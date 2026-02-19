# 📚 ICH QA Evaluation Framework

> Benchmark Large Language Models on **Indian Culture & Heritage** question answering — across multiple prompting strategies, providers, and metrics.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

- **5 prompting strategies** — zero-shot · one-shot · three-shot · five-shot · RAG (BM25)
- **4 evaluation metrics** — Exact Match · Token F1 · ROUGE-L · BERTScore
- **3 API providers** — [Groq](https://groq.com) · [Together AI](https://together.ai) · [Anthropic](https://anthropic.com)
- **2 experiment modes** — with context · without context
- **Per-type breakdown** — results split by factual, boolean, and other questions
- **Crash-safe** — row-level checkpointing with automatic resume on re-run

---

## 🗂 Project Structure

```
ich-qa-eval/
├── config.py            # All settings (provider, models, modes)
├── main.py              # Experiment runner
├── evaluate.py          # Metrics (SQuAD-standard normalization)
├── prompts/
│   ├── __init__.py      # Strategy registry (STRATEGY_BUILDERS)
│   ├── zero_shot.py     # Zero-shot prompt
│   ├── one_shot.py      # 1 example per type
│   ├── three_shot.py    # 3 examples per type
│   ├── five_shot.py     # 5 examples per type
│   └── rag_prompt.py    # BM25-based retrieval
├── data/                # Dataset CSV
├── output/              # Results organized by mode/strategy
├── logs/                # Experiment logs
├── .env.example         # API key template
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/theshivam7/ich-qa-eval.git
cd ich-qa-eval
uv sync
```

<details>
<summary>Using pip instead?</summary>

Use a virtual environment to avoid system package conflicts:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

### 2. Set up your API key

```bash
cp .env.example .env
```

Edit `.env` and add the key for your chosen provider:

```env
# Pick one:
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Choose your provider

Open `config.py` and set `API_PROVIDER`:

```python
API_PROVIDER = "groq"        # gpt-oss-120b
# API_PROVIDER = "together"   # gemma-3-27b, llama-4-maverick
# API_PROVIDER = "anthropic"  # claude-haiku-4-5
```

Models are auto-selected based on your provider — no other changes needed.

### 4. Run

```bash
uv run python main.py
```

---

## 📖 Commands Reference

### Experiment Modes

You can run experiments in two modes:

| Mode | Flag | What the LLM receives | Strategies available |
|------|------|-----------------------|---------------------|
| **With context** (default) | `-m context` | question + context + question_type | all 5 |
| **Without context** | `-m no_context` | question + question_type only | 4 (RAG excluded) |

```bash
# With context (default — no flag needed)
uv run python main.py

# Without context
uv run python main.py -m no_context
```

### All Commands

| What you want | Command |
|---------------|---------|
| Run all strategies **with context** | `uv run python main.py` |
| Run all strategies **without context** | `uv run python main.py -m no_context` |
| Run with 10 samples only | `uv run python main.py -n 10` |
| Run with 50 samples only | `uv run python main.py -n 50` |
| Run only zero-shot | `uv run python main.py -s zero_shot` |
| Run zero-shot + RAG | `uv run python main.py -s zero_shot rag` |
| Run one-shot + three-shot | `uv run python main.py -s one_shot three_shot` |
| 2 strategies, 10 samples | `uv run python main.py -s zero_shot rag -n 10` |
| Without context, 10 samples | `uv run python main.py -m no_context -n 10` |
| Without context, specific strategies | `uv run python main.py -m no_context -s zero_shot one_shot` |

**Strategy names:** `zero_shot` · `one_shot` · `three_shot` · `five_shot` · `rag`

> **Note:** RAG is automatically excluded in `no_context` mode since there's nothing to retrieve from.

### Resuming after interruption

If the run is interrupted (tokens run out, network drops, etc.), just re-run the same command. Progress is saved at the row level — it picks up exactly where it left off.

```bash
# Crashed mid-run? Just re-run:
uv run python main.py

# Or run only the remaining strategies:
uv run python main.py -s three_shot five_shot rag
```

To **re-run** a completed strategy from scratch:

```bash
rm output/context/zero_shot/results_gpt-oss-120b.csv
uv run python main.py -s zero_shot
```

---

## 📊 Dataset Format

Place a CSV in the `data/` directory:

| Column | Description |
|--------|-------------|
| `question` | The question to answer |
| `context` | Source passage |
| `answer` | Ground truth answer |
| `question_type` | `factual`, `boolean`, or `other` |

---

## 📁 Output

Results are organized by mode:

```
output/
├── context/                           # With-context results
│   ├── zero_shot/
│   │   ├── results_gpt-oss-120b.csv
│   │   └── scores_gpt-oss-120b.json
│   ├── one_shot/...
│   ├── three_shot/...
│   ├── five_shot/...
│   ├── rag/...
│   └── comparison_table.csv
│
└── no_context/                   # Without-context results
    ├── zero_shot/
    │   ├── results_gpt-oss-120b.csv
    │   └── scores_gpt-oss-120b.json
    ├── one_shot/...
    ├── three_shot/...
    ├── five_shot/...
    └── comparison_table.csv

logs/experiment.log
```

**Example `scores_gpt-oss-120b.json`:**

```json
{
  "mean_exact_match": 0.215,
  "std_exact_match": 0.411,
  "mean_f1_score": 0.323,
  "std_f1_score": 0.392,
  "mean_rouge_l": 0.322,
  "std_rouge_l": 0.388,
  "mean_bert_score": 0.640,
  "std_bert_score": 0.241,
  "mean_response_time": 1.247,
  "total_prompt_tokens": 52340,
  "total_completion_tokens": 8120
}
```

---

## 📏 Evaluation Metrics

| Metric | Normalization | Description |
|--------|---------------|-------------|
| **Exact Match** | SQuAD | Binary match after normalization (1 or 0) |
| **Token F1** | SQuAD | Word-level precision/recall harmonic mean |
| **ROUGE-L** | Minimal (strip whitespace) | Longest common subsequence F-measure — `rouge_scorer` handles its own tokenization and stemming |
| **BERTScore** | Minimal (strip whitespace) | Semantic similarity via contextual embeddings — operates on natural text for optimal token representations |

**Exact Match** and **Token F1** use **SQuAD normalization** (Rajpurkar et al., 2016):
> lowercase → remove punctuation → remove articles (a, an, the) → collapse whitespace

**ROUGE-L** and **BERTScore** receive minimally cleaned text (whitespace-stripped only), as these metrics have their own internal preprocessing and are designed to operate on natural language.

### Operational Metadata

Each experiment also records per-request operational data, useful for estimating API costs and latency:

| Field | Scope | Description |
|-------|-------|-------------|
| `response_time` | Per row (results CSV) | Wall-clock seconds for the API call |
| `prompt_tokens` | Per row (results CSV) | Input tokens consumed |
| `completion_tokens` | Per row (results CSV) | Output tokens generated |
| `mean_response_time` | Aggregate (scores JSON, comparison table) | Average response time across all rows |
| `total_prompt_tokens` | Aggregate (scores JSON, comparison table) | Sum of input tokens for the experiment |
| `total_completion_tokens` | Aggregate (scores JSON, comparison table) | Sum of output tokens for the experiment |

---

## ⚙️ Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `API_PROVIDER` | `"groq"` | API provider to use |
| `MAX_TOKENS` | `1024` | Max LLM response tokens |
| `TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `MAX_RETRIES` | `5` | API retry attempts |
| `RETRY_BACKOFF_BASE` | `2` | Exponential backoff (seconds) |
| `RAG_TOP_K` | `3` | Number of chunks to retrieve |
| `RAG_CHUNK_SIZE` | `200` | Words per chunk |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
