# ICH QA Evaluation Framework

Evaluate how well Large Language Models answer questions about **Indian Culture & Heritage**. This framework tests multiple models with different prompting strategies and measures accuracy using standard NLP metrics.

## What It Does

- Tests LLMs using 5 prompting strategies (zero-shot, one-shot, three-shot, five-shot, RAG)
- Computes 4 metrics: Exact Match, Token F1, ROUGE-L, BERTScore
- Breaks down results by question type (factual, boolean, other)
- Generates a comparison table across all experiments
- Supports 3 API providers: [Groq](https://groq.com), [Together AI](https://together.ai), [Anthropic](https://anthropic.com)

## Project Structure

```
ich-qa-eval/
├── config.py            # All settings (provider, models, paths)
├── main.py              # Runs experiments
├── evaluate.py          # Computes metrics
├── prompts/
│   ├── zero_shot.py
│   ├── one_shot.py
│   ├── three_shot.py
│   ├── five_shot.py
│   └── rag_prompt.py    # BM25-based retrieval
├── data/                # Your dataset goes here
├── .env.example         # API key template
├── requirements.txt
└── pyproject.toml
```

## Setup

### 1. Install

```bash
git clone https://github.com/theshivam7/ich-qa-eval.git
cd ich-qa-eval
pip install -r requirements.txt
```

### 2. Add your API key

Copy the example and fill in your key:

```bash
cp .env.example .env
```

Then edit `.env` — you only need the key for the provider you're using:

```
GROQ_API_KEY=your_key_here
```

### 3. Pick your provider

Open `config.py` and set `API_PROVIDER` (line 20):

```python
API_PROVIDER = "groq"       # → uses GROQ_API_KEY, runs gpt-oss-120b
# API_PROVIDER = "together"  # → uses TOGETHER_API_KEY, runs gemma-3-27b, llama-4-maverick
# API_PROVIDER = "anthropic" # → uses ANTHROPIC_API_KEY, runs claude-haiku-4-5
```

Each provider has its own set of models. They are automatically selected — no other changes needed.

### 4. Run

```bash
python main.py
```

## Dataset Format

Place a CSV in the `data/` directory with these columns:

| Column | Description |
|--------|-------------|
| `question` | The question |
| `context` | Source passage |
| `answer` | Ground truth answer |
| `question_type` | `factual`, `boolean`, or `other` |

## Output

After running, you'll find:

```
output/
├── zero_shot/results_gpt-oss-120b.csv
├── one_shot/results_gpt-oss-120b.csv
├── ...
└── comparison_table.csv          # All metrics in one table
logs/experiment.log               # Full run log
```

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **Exact Match** | Does the answer exactly match? (after normalization) |
| **Token F1** | Word-level overlap between prediction and reference |
| **ROUGE-L** | Longest common subsequence similarity |
| **BERTScore** | Semantic similarity using contextual embeddings |

## Configuration

Key settings in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `API_PROVIDER` | `"groq"` | Which API to use |
| `MAX_TOKENS` | `1024` | Max response length |
| `MAX_RETRIES` | `5` | Retry attempts on API failure |
| `RAG_TOP_K` | `3` | Chunks to retrieve for RAG |
| `RAG_CHUNK_SIZE` | `200` | Words per chunk |

## License

[MIT](LICENSE)
