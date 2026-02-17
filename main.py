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
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content
    return answer.strip() if answer else ""


def _call_anthropic(client, model_id, prompt):
    """Call the Anthropic Messages API."""
    message = client.messages.create(
        model=model_id,
        max_tokens=config.MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    if not message.content:
        return ""
    answer = message.content[0].text
    return answer.strip() if answer else ""


def call_llm_api(client, model_id, prompt, logger):
    """Call the LLM API with exponential backoff retry."""
    is_anthropic = config.API_PROVIDER == "anthropic"

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
                return ""
            time.sleep(wait_time)

    return ""


def run_experiment(client, model_name, model_id, strategy_name, df, logger):
    """Run a single (model, strategy) experiment. Returns DataFrame with llm_answer column."""
    prompt_builder = STRATEGY_BUILDERS[strategy_name]
    llm_answers = []
    total = len(df)

    logger.info(
        "Starting: model=%s, strategy=%s, rows=%d",
        model_name, strategy_name, total,
    )

    for row_num, (_, row) in enumerate(df.iterrows(), start=1):
        prompt = prompt_builder(
            str(row["question"]), str(row["context"]), str(row["question_type"])
        )
        llm_answers.append(call_llm_api(client, model_id, prompt, logger))

        if row_num % 10 == 0 or row_num == total:
            logger.info("  Progress: %d/%d", row_num, total)

    results_df = df.copy()
    results_df["llm_answer"] = llm_answers
    return results_df


def compute_metrics_for_results(results_df, logger):
    """Compute metrics overall and per question_type."""
    records = []

    overall = compute_all_metrics(
        results_df["llm_answer"].tolist(),
        results_df["answer"].tolist(),
        results_df["question_type"].tolist(),
    )
    overall["question_type"] = "overall"
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
        records.append(qt_metrics)
        logger.info("  Metrics for '%s' (n=%d)", qt, qt_metrics["count"])

    return records


def save_experiment_results(results_df, model_name, strategy_name, logger):
    """Save per-row results CSV to output/<strategy>/."""
    output_dir = config.STRATEGY_OUTPUT_DIRS[strategy_name]
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model = model_name.replace("/", "_").replace(" ", "_")
    filepath = output_dir / f"results_{safe_model}.csv"

    results_df.to_csv(filepath, index=False)
    logger.info("Saved results to %s", filepath)


def save_metric_scores_json(metric_records, model_name, strategy_name, logger):
    """Save overall metric scores as a JSON file to output/<strategy>/."""
    output_dir = config.STRATEGY_OUTPUT_DIRS[strategy_name]
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
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("ICH QA Evaluation Framework - Starting")
    logger.info("=" * 80)

    load_dotenv()
    client = build_client(logger)

    try:
        df = load_dataset()
        logger.info(
            "Dataset loaded: %d rows, types: %s",
            len(df), df["question_type"].value_counts().to_dict(),
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    for d in config.STRATEGY_OUTPUT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    total_experiments = len(config.MODELS) * len(config.PROMPT_STRATEGIES)
    experiment_num = 0

    for model_name, model_id in config.MODELS.items():
        for strategy_name in config.PROMPT_STRATEGIES:
            experiment_num += 1
            logger.info("-" * 60)
            logger.info(
                "Experiment %d/%d: model=%s, strategy=%s",
                experiment_num, total_experiments, model_name, strategy_name,
            )
            logger.info("-" * 60)

            results_df = run_experiment(
                client, model_name, model_id, strategy_name, df, logger
            )
            save_experiment_results(results_df, model_name, strategy_name, logger)

            logger.info("Computing metrics...")
            metric_records = compute_metrics_for_results(results_df, logger)
            save_metric_scores_json(metric_records, model_name, strategy_name, logger)
            for record in metric_records:
                record["model"] = model_name
                record["strategy"] = strategy_name
                all_metrics.append(record)

            logger.info("Completed: model=%s, strategy=%s", model_name, strategy_name)

    logger.info("Building final comparison table...")
    comparison_df = build_comparison_table(all_metrics)

    comparison_path = config.OUTPUT_DIR / "comparison_table.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Comparison table saved to %s", comparison_path)

    print_comparison_table(comparison_df, logger)

    logger.info("=" * 80)
    logger.info("ICH QA Evaluation Framework - Complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
