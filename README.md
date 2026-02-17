# 📚 ICH QA Evaluation Framework

> Benchmark Large Language Models on **Indian Culture & Heritage** question answering — across multiple prompting strategies, providers, and metrics.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

- **5 prompting strategies** — zero-shot · one-shot · three-shot · five-shot · RAG (BM25)
- **4 evaluation metrics** — Exact Match · Token F1 · ROUGE-L · BERTScore
- **3 API providers** — [Groq](https://groq.com) · [Together AI](https://together.ai) · [Anthropic](https://anthropic.com)
- **Per-type breakdown** — results split by factual, boolean, and other questions
- **Crash-safe** — row-level checkpointing with automatic resume on re-run

---

## 🗂 Project Structure

```
ich-qa-eval/
├── config.py            # All settings (provider, models, paths)
├── main.py              # Experiment runner
├── evaluate.py          # Metrics (SQuAD-standard normalization)
├── prompts/
│   ├── zero_shot.py     # Zero-shot prompt
│   ├── one_shot.py      # 1 example per type
│   ├── three_shot.py    # 3 examples per type
│   ├── five_shot.py     # 5 examples per type
│   └── rag_prompt.py    # BM25-based retrieval
├── data/                # Dataset CSV
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

```bash
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

| What you want | Command |
|---------------|---------|
| Run all strategies, all samples | `uv run python main.py` |
| Run with 10 samples only | `uv run python main.py -n 10` |
| Run with 50 samples only | `uv run python main.py -n 50` |
| Run only zero-shot | `uv run python main.py -s zero_shot` |
| Run zero-shot + RAG | `uv run python main.py -s zero_shot rag` |
| Run one-shot + three-shot | `uv run python main.py -s one_shot three_shot` |
| Run 2 strategies on 10 samples | `uv run python main.py -s zero_shot rag -n 10` |

**Strategy names:** `zero_shot` · `one_shot` · `three_shot` · `five_shot` · `rag`

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
rm output/zero_shot/results_gpt-oss-120b.csv
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

```
output/
├── zero_shot/
│   ├── results_gpt-oss-120b.csv      # Per-row LLM answers
│   └── scores_gpt-oss-120b.json      # Overall metric scores
├── one_shot/...
├── three_shot/...
├── five_shot/...
├── rag/...
└── comparison_table.csv               # All metrics in one table

logs/experiment.log                    # Full run log
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
  "std_bert_score": 0.241
}
```

---

## 📏 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match** | Binary match after normalization (1 or 0) |
| **Token F1** | Word-level precision/recall harmonic mean |
| **ROUGE-L** | Longest common subsequence F-measure |
| **BERTScore** | Semantic similarity via contextual embeddings |

All text normalization follows the **SQuAD evaluation standard** (Rajpurkar et al., 2016):
> lowercase → remove punctuation → remove articles (a, an, the) → collapse whitespace

---

## ⚙️ Configuration

All settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `API_PROVIDER` | `"groq"` | API provider to use |
| `MAX_TOKENS` | `1024` | Max LLM response tokens |
| `MAX_RETRIES` | `5` | API retry attempts |
| `RETRY_BACKOFF_BASE` | `2` | Exponential backoff (seconds) |
| `RAG_TOP_K` | `3` | Number of chunks to retrieve |
| `RAG_CHUNK_SIZE` | `200` | Words per chunk |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
