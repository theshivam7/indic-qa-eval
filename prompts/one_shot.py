"""
one_shot.py - One-shot prompt template (1 example per category).
"""

from prompts.zero_shot import SYSTEM_INSTRUCTION

FACTOID_EXAMPLE = """
Example (Factoid):
Context: Matsya (Sanskrit: Matsya; Pāli: Macchā) was an ancient Indo-Aryan tribe of central South Asia whose existence is attested during the Iron Age. Macchā in Pāli and Matsya in Sanskrit mean fish. The kingdom covered an extensive territory, with the Sarasvatī river as its western border.
Question Type: factual
Question: What does Macchā in Pāli and Matsya in Sanskrit mean?
Answer: fish
"""

BOOLEAN_EXAMPLE = """
Example (Boolean):
Context: La Martiniere Calcutta is a private school established in 1836 by Claude Martin. It follows the ICSE curriculum and is managed by a private trust.
Question Type: boolean
Question: Is La Martiniere Calcutta a government school?
Answer: No
"""

OTHERS_EXAMPLE = """
Example (Other):
Context: Fort Geldria or Fort Geldaria, located in Pulicat, was the seat of the Dutch Republic's first settlement in India, and the capital of Dutch Coromandel. It was built in 1613 and served as a strategic trading post for spice trade.
Question Type: other
Question: Explain the significance of Fort Geldria in Pulicat.
Answer: Fort Geldria or Fort Geldaria, located in Pulicat, was the seat of the Dutch Republic's first settlement in India, and the capital of Dutch Coromandel.
"""


def build_prompt(question, context, question_type, **kwargs):
    """Build a one-shot prompt with 1 example per category."""
    examples = FACTOID_EXAMPLE + BOOLEAN_EXAMPLE + OTHERS_EXAMPLE
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Here are examples showing the expected answer format:\n"
        f"{examples}\n"
        f"Now answer the following:\n\n"
        f"Context:\n{context}\n\n"
        f"Question Type: {question_type}\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
