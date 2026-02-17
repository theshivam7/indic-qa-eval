"""
evaluate.py - Metric computation: Exact Match, F1, ROUGE-L, BERTScore.
"""

import re
import string
import logging
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn

logger = logging.getLogger(__name__)

_rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def normalize_text(text):
    """Normalize text following SQuAD evaluation standard (Rajpurkar et al., 2016).

    Steps: lowercase → remove punctuation → remove articles → collapse whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles (standard SQuAD normalization)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_boolean(text):
    """Map boolean answer variations to 'yes' or 'no'."""
    if not isinstance(text, str):
        text = str(text)
    # Check for 'Answer: yes/no' pattern before stripping punctuation
    answer_match = re.search(r"[Aa]nswer\s*:\s*(yes|no)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).lower()
    text = normalize_text(text)
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def extract_answer(text, question_type):
    """Extract the answer portion from model output (after 'Answer:')."""
    if not isinstance(text, str):
        text = str(text)

    answer_match = re.search(r"[Aa]nswer\s*:\s*(.+)", text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        if question_type == "boolean":
            return answer_text.split("\n")[0].strip()
        # Take only the first paragraph for factoid/other to avoid trailing noise
        paragraphs = answer_text.split("\n\n")
        return paragraphs[0].strip()

    return text.strip()


def exact_match(prediction, reference, question_type="factual"):
    """Return 1.0 if normalized prediction equals reference, else 0.0."""
    if question_type == "boolean":
        pred_norm = normalize_boolean(prediction)
        ref_norm = normalize_boolean(reference)
    else:
        pred_norm = normalize_text(prediction)
        ref_norm = normalize_text(reference)
    return 1.0 if pred_norm == ref_norm else 0.0


def f1_score_tokens(prediction, reference, question_type="factual"):
    """Token-level F1 using Counter for multi-occurrence overlap."""
    if question_type == "boolean":
        pred_tokens = normalize_boolean(prediction).split()
        ref_tokens = normalize_boolean(reference).split()
    else:
        pred_tokens = normalize_text(prediction).split()
        ref_tokens = normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    num_common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l_score(prediction, reference, question_type="factual"):
    """Compute ROUGE-L F1 score."""
    if question_type == "boolean":
        pred = normalize_boolean(prediction)
        ref = normalize_boolean(reference)
    else:
        pred = normalize_text(prediction)
        ref = normalize_text(reference)

    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0

    return _rouge_scorer.score(ref, pred)["rougeL"].fmeasure


def bert_score_value(predictions, references):
    """Compute BERTScore F1 for a batch. Returns list of floats."""
    if not predictions:
        return []
    try:
        _p, _r, f1 = bert_score_fn(
            predictions, references,
            lang="en", verbose=False, rescale_with_baseline=True,
        )
        return f1.tolist()
    except Exception as e:
        logger.error("BERTScore failed: %s", e)
        return [0.0] * len(predictions)


def compute_all_metrics(predictions, references, question_types):
    """Compute mean/std for all metrics over prediction-reference pairs."""
    n = len(predictions)
    if n == 0:
        return {
            "mean_exact_match": 0.0, "std_exact_match": 0.0,
            "mean_f1_score": 0.0, "std_f1_score": 0.0,
            "mean_rouge_l": 0.0, "std_rouge_l": 0.0,
            "mean_bert_score": 0.0, "std_bert_score": 0.0,
            "count": 0,
        }

    extracted_preds = [
        extract_answer(p, qt) for p, qt in zip(predictions, question_types)
    ]

    em_scores = []
    f1_scores = []
    rl_scores = []
    for pred, ref, qt in zip(extracted_preds, references, question_types):
        em_scores.append(exact_match(pred, ref, qt))
        f1_scores.append(f1_score_tokens(pred, ref, qt))
        rl_scores.append(rouge_l_score(pred, ref, qt))

    # Normalize inputs for BERTScore
    bert_preds, bert_refs = [], []
    for pred, ref, qt in zip(extracted_preds, references, question_types):
        if qt == "boolean":
            bert_preds.append(normalize_boolean(pred))
            bert_refs.append(normalize_boolean(ref))
        else:
            bert_preds.append(normalize_text(pred))
            bert_refs.append(normalize_text(ref))

    bs_scores = bert_score_value(bert_preds, bert_refs)

    return {
        "mean_exact_match": float(np.mean(em_scores)),
        "std_exact_match": float(np.std(em_scores)),
        "mean_f1_score": float(np.mean(f1_scores)),
        "std_f1_score": float(np.std(f1_scores)),
        "mean_rouge_l": float(np.mean(rl_scores)),
        "std_rouge_l": float(np.std(rl_scores)),
        "mean_bert_score": float(np.mean(bs_scores)) if bs_scores else 0.0,
        "std_bert_score": float(np.std(bs_scores)) if bs_scores else 0.0,
        "count": n,
    }
