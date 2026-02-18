"""
main.py - Experiment runner for the ICH QA Evaluation Framework.

Runs each (model, prompting_strategy) combination, calls the configured
LLM API (Together AI, Groq, or Anthropic), computes metrics, and
generates a comparison table.
"""

import os
import sys
import time
import json
import logging
import argparse

import pandas as pd
from dotenv import load_dotenv

import config
from evaluate import compute_all_metrics
from prompts import STRATEGY_BUILDERS


def setup_logging():
    """Configure logging to file and console."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOG_DIR / config.LOG_FILENAME

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger(__name__)


def build_client(logger):
    """Initialize the API client based on API_PROVIDER in config."""
    provider = config.API_PROVIDER

    if provider == "together":
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            logger.error("TOGETHER_API_KEY not found. Add it to your .env file.")
            sys.exit(1)
        from together import Together
        client = Together(api_key=api_key)

    elif provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found. Add it to your .env file.")
            sys.exit(1)
        from groq import Groq
        client = Groq(api_key=api_key)

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("ANTHROPIC_API_KEY not found. Add it to your .env file.")
            sys.exit(1)
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

    else:
        logger.error("Unknown API_PROVIDER: '%s'. Use 'together', 'groq', or 'anthropic'.", provider)
        sys.exit(1)

    logger.info("API client initialized (provider=%s).", provider)
    return client


def load_dataset():
    """Load and validate the QA dataset CSV from data/."""
    csv_path = config.DATA_DIR / config.DATASET_FILENAME

    if not csv_path.exists():
        try:
            csv_files = [f.name for f in config.DATA_DIR.iterdir() if f.suffix == ".csv"]
        except FileNotFoundError:
            csv_files = []

        if csv_files:
            csv_path = config.DATA_DIR / csv_files[0]
            logging.getLogger(__name__).info("Using dataset file: %s", csv_files[0])
        else:
            raise FileNotFoundError(
                f"No CSV found in {config.DATA_DIR}. "
                f"Place your dataset as '{config.DATASET_FILENAME}' in data/."
            )

    df = pd.read_csv(csv_path)

    required_cols = {"question", "context", "answer", "question_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df["question_type"] = df["question_type"].str.strip().str.lower()
    df["question"] = df["question"].fillna("")
    df["context"] = df["context"].fillna("")
    df["answer"] = df["answer"].fillna("")

    return df


def _call_openai_compatible(client, model_id, prompt):
    """Call an OpenAI-compatible API (Together AI, Groq)."""
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_id,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start
    answer = response.choices[0].message.content
    usage = response.usage
    return {
        "answer": answer.strip() if answer else "",
        "response_time": round(elapsed, 3),
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
    }


def _call_anthropic(client, model_id, prompt):
    """Call the Anthropic Messages API."""
    start = time.perf_counter()
    message = client.messages.create(
        model=model_id,
        max_tokens=config.MAX_TOKENS,
        temperature=config.TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.perf_counter() - start
    usage = message.usage
    answer = message.content[0].text if message.content else ""
    return {
        "answer": answer.strip() if answer else "",
        "response_time": round(elapsed, 3),
        "prompt_tokens": usage.input_tokens if usage else 0,
        "completion_tokens": usage.output_tokens if usage else 0,
    }


def call_llm_api(client, model_id, prompt, logger):
    """Call the LLM API with exponential backoff retry.

    Returns a dict with keys: answer, response_time, prompt_tokens, completion_tokens.
    """
    is_anthropic = config.API_PROVIDER == "anthropic"
    empty_result = {"answer": "", "response_time": 0.0, "prompt_tokens": 0, "completion_tokens": 0}

    for attempt in range(1, config.MAX_RETRIES + 1):
        try:
            if is_anthropic:
                return _call_anthropic(client, model_id, prompt)
            return _call_openai_compatible(client, model_id, prompt)

        except Exception as e:
            wait_time = config.RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "API failed (attempt %d/%d): %s. Retry in %ds...",
                attempt, config.MAX_RETRIES, str(e), wait_time,
            )
            if attempt == config.MAX_RETRIES:
                logger.error("Max retries reached. Returning empty answer.")
                return empty_result
            time.sleep(wait_time)

    return empty_result


def run_experiment(client, model_name, model_id, strategy_name, df,
                    strategy_output_dirs, logger):
    """Run a single (model, strategy) experiment with row-level checkpointing.

    Saves each answer incrementally to a .partial.csv file. If interrupted,
    re-running will resume from the last saved row.
    """
    prompt_builder = STRATEGY_BUILDERS[strategy_name]
    total = len(df)

    # Checkpoint path: output/<mode>/<strategy>/results_<model>.partial.csv
    safe_model = model_name.replace("/", "_").replace(" ", "_")
    output_dir = strategy_output_dirs[strategy_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"results_{safe_model}.partial.csv"

    # Resume from checkpoint if it exists and is valid
    start_row = 0
    llm_answers = []
    response_times = []
    prompt_tokens_list = []
    completion_tokens_list = []
    if checkpoint_path.exists():
        try:
            partial_df = pd.read_csv(checkpoint_path)
            saved_rows = len(partial_df)
            # Validate: checkpoint must not exceed current dataset size
            if saved_rows <= total:
                start_row = saved_rows
                # Use astype(str) to prevent NaN/float type issues
                llm_answers = partial_df["llm_answer"].fillna("").astype(str).tolist()
                # Gracefully handle old checkpoints that lack timing/token columns
                if "response_time" in partial_df.columns:
                    response_times = partial_df["response_time"].fillna(0.0).tolist()
                else:
                    response_times = [0.0] * saved_rows
                if "prompt_tokens" in partial_df.columns:
                    prompt_tokens_list = partial_df["prompt_tokens"].fillna(0).astype(int).tolist()
                else:
                    prompt_tokens_list = [0] * saved_rows
                if "completion_tokens" in partial_df.columns:
                    completion_tokens_list = partial_df["completion_tokens"].fillna(0).astype(int).tolist()
                else:
                    completion_tokens_list = [0] * saved_rows
                logger.info(
                    "Resuming from checkpoint: %d/%d rows already done.",
                    start_row, total,
                )
            else:
                logger.warning(
                    "Stale checkpoint (%d rows) exceeds dataset (%d rows). Starting fresh.",
                    saved_rows, total,
                )
                checkpoint_path.unlink()
        except Exception as e:
            logger.warning("Corrupt checkpoint, starting fresh: %s", e)
            checkpoint_path.unlink()

    logger.info(
        "Starting: model=%s, strategy=%s, rows=%d (from row %d)",
        model_name, strategy_name, total, start_row + 1,
    )

    for row_num, (_, row) in enumerate(df.iterrows()):
        if row_num < start_row:
            continue  # already answered

        prompt = prompt_builder(
            str(row["question"]), str(row["context"]), str(row["question_type"])
        )
        result = call_llm_api(client, model_id, prompt, logger)
        llm_answers.append(result["answer"])
        response_times.append(result["response_time"])
        prompt_tokens_list.append(result["prompt_tokens"])
        completion_tokens_list.append(result["completion_tokens"])

        # Save checkpoint after every row
        checkpoint_df = df.head(len(llm_answers)).copy()
        checkpoint_df["llm_answer"] = llm_answers
        checkpoint_df["response_time"] = response_times
        checkpoint_df["prompt_tokens"] = prompt_tokens_list
        checkpoint_df["completion_tokens"] = completion_tokens_list
        checkpoint_df.to_csv(checkpoint_path, index=False)

        if (row_num + 1) % 10 == 0 or row_num + 1 == total:
            logger.info("  Progress: %d/%d", row_num + 1, total)

    results_df = df.copy()
    results_df["llm_answer"] = llm_answers
    results_df["response_time"] = response_times
    results_df["prompt_tokens"] = prompt_tokens_list
    results_df["completion_tokens"] = completion_tokens_list

    # Remove checkpoint after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return results_df


def _add_timing_token_stats(record, subset_df):
    """Add response time and token usage stats to a metric record."""
    record["mean_response_time"] = round(subset_df["response_time"].mean(), 3)
    record["total_prompt_tokens"] = int(subset_df["prompt_tokens"].sum())
    record["total_completion_tokens"] = int(subset_df["completion_tokens"].sum())


def compute_metrics_for_results(results_df, logger):
    """Compute metrics overall and per question_type."""
    records = []

    overall = compute_all_metrics(
        results_df["llm_answer"].tolist(),
        results_df["answer"].tolist(),
        results_df["question_type"].tolist(),
    )
    overall["question_type"] = "overall"
    _add_timing_token_stats(overall, results_df)
    records.append(overall)
    logger.info("  Overall metrics computed (n=%d)", overall["count"])

    for qt in sorted(results_df["question_type"].unique()):
        subset = results_df[results_df["question_type"] == qt]
        if len(subset) == 0:
            continue
        qt_metrics = compute_all_metrics(
            subset["llm_answer"].tolist(),
            subset["answer"].tolist(),
            subset["question_type"].tolist(),
        )
        qt_metrics["question_type"] = qt
        _add_timing_token_stats(qt_metrics, subset)
        records.append(qt_metrics)
        logger.info("  Metrics for '%s' (n=%d)", qt, qt_metrics["count"])

    return records


def save_experiment_results(results_df, model_name, strategy_name,
                            strategy_output_dirs, logger):
    """Save per-row results CSV to output/<mode>/<strategy>/."""
    output_dir = strategy_output_dirs[strategy_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace("/", "_").replace(" ", "_")
    filepath = output_dir / f"results_{safe_model}.csv"

    results_df.to_csv(filepath, index=False)
    logger.info("Saved results to %s", filepath)


def save_metric_scores_json(metric_records, model_name, strategy_name,
                            strategy_output_dirs, logger):
    """Save overall metric scores as a JSON file to output/<mode>/<strategy>/."""
    output_dir = strategy_output_dirs[strategy_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find the 'overall' record
    overall = next((r for r in metric_records if r.get("question_type") == "overall"), None)
    if not overall:
        return

    scores = {
        "mean_exact_match": overall["mean_exact_match"],
        "std_exact_match": overall["std_exact_match"],
        "mean_f1_score": overall["mean_f1_score"],
        "std_f1_score": overall["std_f1_score"],
        "mean_rouge_l": overall["mean_rouge_l"],
        "std_rouge_l": overall["std_rouge_l"],
        "mean_bert_score": overall["mean_bert_score"],
        "std_bert_score": overall["std_bert_score"],
        "mean_response_time": overall["mean_response_time"],
        "total_prompt_tokens": overall["total_prompt_tokens"],
        "total_completion_tokens": overall["total_completion_tokens"],
    }

    safe_model = model_name.replace("/", "_").replace(" ", "_")
    filepath = output_dir / f"scores_{safe_model}.json"

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    logger.info("Saved scores to %s", filepath)


def build_comparison_table(all_metrics):
    """Build a sorted DataFrame from all experiment metrics."""
    comparison_df = pd.DataFrame(all_metrics)

    ordered_cols = [
        "model", "strategy", "question_type", "count",
        "mean_exact_match", "std_exact_match",
        "mean_f1_score", "std_f1_score",
        "mean_rouge_l", "std_rouge_l",
        "mean_bert_score", "std_bert_score",
        "mean_response_time", "total_prompt_tokens", "total_completion_tokens",
    ]
    ordered_cols = [c for c in ordered_cols if c in comparison_df.columns]
    return comparison_df[ordered_cols]


def print_comparison_table(comparison_df, logger):
    """Print comparison table to console and log."""
    separator = "=" * 120
    logger.info(separator)
    logger.info("COMPARISON TABLE")
    logger.info(separator)

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.float_format", "{:.4f}".format,
    ):
        for line in comparison_df.to_string(index=False).split("\n"):
            logger.info(line)

    logger.info(separator)


def main():
    """Run all experiments and generate the comparison table."""
    parser = argparse.ArgumentParser(description="ICH QA Evaluation Framework")
    parser.add_argument(
        "-n", "--samples", type=int, default=None,
        help="Number of samples to evaluate (default: all rows)",
    )
    parser.add_argument(
        "-s", "--strategies", nargs="+", default=None,
        help="Prompt strategies to run (default: all for the mode)",
    )
    parser.add_argument(
        "-m", "--mode", choices=config.VALID_MODES, default="context",
        help="Experiment mode: 'context' (default) or 'no_context'",
    )
    args = parser.parse_args()

    mode = args.mode

    # Determine strategies for this mode
    available_strategies = (
        config.NO_CONTEXT_STRATEGIES if mode == "no_context"
        else config.PROMPT_STRATEGIES
    )
    if args.strategies:
        invalid = [s for s in args.strategies if s not in available_strategies]
        if invalid:
            parser.error(
                f"Invalid strategies for mode '{mode}': {invalid}. "
                f"Available: {available_strategies}"
            )
        strategies = args.strategies
    else:
        strategies = available_strategies

    # Build output dirs for this mode
    strategy_output_dirs = config.get_strategy_output_dirs(mode)

    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("ICH QA Evaluation Framework - Starting (mode=%s)", mode)
    logger.info("=" * 80)

    load_dotenv()

    if not config.MODELS:
        logger.error(
            "No models configured for API_PROVIDER='%s'. "
            "Check PROVIDER_MODELS in config.py.",
            config.API_PROVIDER,
        )
        sys.exit(1)

    client = build_client(logger)

    try:
        df = load_dataset()
        if args.samples:
            df = df.head(args.samples)
            logger.info("Using first %d samples.", args.samples)
        logger.info(
            "Dataset loaded: %d rows, types: %s",
            len(df), df["question_type"].value_counts().to_dict(),
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    # In no_context mode, blank out the context column
    if mode == "no_context":
        df["context"] = ""
        logger.info("Mode=no_context: context column cleared.")

    for d in strategy_output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    total_experiments = len(config.MODELS) * len(strategies)
    experiment_num = 0
    mode_output_dir = config.OUTPUT_DIR / config.MODE_DIR_NAMES[mode]
    comparison_path = mode_output_dir / "comparison_table.csv"

    for model_name, model_id in config.MODELS.items():
        for strategy_name in strategies:
            experiment_num += 1

            # Skip if final results already exist (fully completed)
            safe_model = model_name.replace("/", "_").replace(" ", "_")
            existing_csv = strategy_output_dirs[strategy_name] / f"results_{safe_model}.csv"
            if existing_csv.exists():
                logger.info(
                    "Skipping %d/%d (already exists): model=%s, strategy=%s",
                    experiment_num, total_experiments, model_name, strategy_name,
                )
                continue

            logger.info("-" * 60)
            logger.info(
                "Experiment %d/%d: model=%s, strategy=%s",
                experiment_num, total_experiments, model_name, strategy_name,
            )
            logger.info("-" * 60)

            try:
                results_df = run_experiment(
                    client, model_name, model_id, strategy_name, df,
                    strategy_output_dirs, logger,
                )
                save_experiment_results(
                    results_df, model_name, strategy_name,
                    strategy_output_dirs, logger,
                )

                logger.info("Computing metrics...")
                metric_records = compute_metrics_for_results(results_df, logger)
                save_metric_scores_json(
                    metric_records, model_name, strategy_name,
                    strategy_output_dirs, logger,
                )
                for record in metric_records:
                    record["model"] = model_name
                    record["strategy"] = strategy_name
                    all_metrics.append(record)

                # Save comparison table after each experiment (crash-safe)
                if all_metrics:
                    comparison_df = build_comparison_table(all_metrics)
                    comparison_df.to_csv(comparison_path, index=False)

                logger.info("Completed: model=%s, strategy=%s", model_name, strategy_name)

            except Exception as e:
                logger.error(
                    "Experiment failed (model=%s, strategy=%s): %s",
                    model_name, strategy_name, e,
                )
                logger.info("Continuing with next experiment...")

    if all_metrics:
        comparison_df = build_comparison_table(all_metrics)
        comparison_df.to_csv(comparison_path, index=False)
        logger.info("Comparison table saved to %s", comparison_path)
        print_comparison_table(comparison_df, logger)
    else:
        logger.info("No new experiments were run.")

    logger.info("=" * 80)
    logger.info("ICH QA Evaluation Framework - Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

