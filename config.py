"""
config.py - Central configuration for the ICH QA Evaluation Framework.
"""

from pathlib import Path

# Directory paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"

# ── API Provider ──────────────────────────────────────────────
# Switch between providers: "together", "groq", or "anthropic"
# Set the matching API key in your .env file:
#   "together"  → TOGETHER_API_KEY
#   "groq"      → GROQ_API_KEY
#   "anthropic" → ANTHROPIC_API_KEY
API_PROVIDER = "groq"

# ── Models per provider ──────────────────────────────────────
# Only the models for the active API_PROVIDER are used.
# Add/remove model entries under the provider you use.

PROVIDER_MODELS = {
    "groq": {
        "gpt-oss-120b": "openai/gpt-oss-120b",
    },
    "together": {
        "gemma-3-27b": "google/gemma-3-27b-it",
        "llama-4-maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
    "anthropic": {
        "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    },
}

# Active models — automatically selected from the provider above
MODELS = PROVIDER_MODELS.get(API_PROVIDER, {})

# Max tokens for LLM response (used by Anthropic; Together/Groq use model defaults)
MAX_TOKENS = 1024

# Prompts
PROMPT_STRATEGIES = [
    "zero_shot",
    "one_shot",
    "three_shot",
    "five_shot",
    "rag",
]

STRATEGY_OUTPUT_DIRS = {s: OUTPUT_DIR / s for s in PROMPT_STRATEGIES}

# Question types (must match the values in the dataset CSV)
QUESTION_TYPES = ["factual", "boolean", "other"]

# API retry settings
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2  # exponential backoff base in seconds

# RAG settings
RAG_TOP_K = 3
RAG_CHUNK_SIZE = 200  # words

# File names
DATASET_FILENAME = "qa_test_human_200.csv"
LOG_FILENAME = "experiment.log"
