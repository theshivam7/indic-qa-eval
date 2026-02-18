"""
rag_prompt.py - RAG-based prompt using BM25 retrieval over context chunks.
"""

import config
from rank_bm25 import BM25Okapi


FACTOID_INSTRUCTION = (
    "Provide a direct, concise, factual answer from the context."
)

BOOLEAN_INSTRUCTION = (
    "Answer with exactly 'Yes' or 'No'. Do not add any explanation."
)

OTHERS_INSTRUCTION = (
    "Provide a thorough but concise answer based strictly on the context."
)

QUESTION_TYPE_INSTRUCTIONS = {
    "factual": FACTOID_INSTRUCTION,
    "boolean": BOOLEAN_INSTRUCTION,
    "other": OTHERS_INSTRUCTION,
}


def chunk_context(context, chunk_size=None):
    """Split context into word-level chunks."""
    if not context or not context.strip():
        return []

    if chunk_size is None:
        chunk_size = config.RAG_CHUNK_SIZE

    words = context.split()
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def retrieve_top_k(question, context, top_k=None):
    """Use BM25 to retrieve the top-k most relevant chunks."""
    if top_k is None:
        top_k = config.RAG_TOP_K

    chunks = chunk_context(context)
    if not chunks:
        return [context] if context and context.strip() else [""]

    if len(chunks) <= top_k:
        return chunks

    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(question.lower().split())

    # Inner sort: pick top-k by BM25 score; outer sort: restore original document order
    top_indices = sorted(
        sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    )
    return [chunks[i] for i in top_indices]


def build_prompt(question, context, question_type, **kwargs):
    """Build a RAG prompt with retrieved passages and direct answering style."""
    passages = retrieve_top_k(question, context)
    retrieved_context = "\n\n".join(passages)

    type_instruction = QUESTION_TYPE_INSTRUCTIONS.get(question_type, FACTOID_INSTRUCTION)

    prompt = (
        f"You are given a context passage and a question about Indian culture and history. "
        f"Answer the question based strictly on the information provided in the context.\n\n"
        f"Provide a direct, concise answer. Do not include explanations, reasoning, "
        f"or information not present in the context.\n\n"
        f"{type_instruction}\n\n"
        f"Context:\n{retrieved_context}\n\n"
        f"Question Type: {question_type}\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    return prompt
