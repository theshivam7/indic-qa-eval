"""
five_shot.py - Five-shot prompt template (5 examples per category).
"""

from prompts.zero_shot import SYSTEM_INSTRUCTION_WITH_CONTEXT, SYSTEM_INSTRUCTION_NO_CONTEXT

FACTOID_EXAMPLES = """
Example 1 (Factoid):
Context: Matsya (Sanskrit: Matsya; Pāli: Macchā) was an ancient Indo-Aryan tribe of central South Asia whose existence is attested during the Iron Age. Macchā in Pāli and Matsya in Sanskrit mean fish. The kingdom covered an extensive territory, with the Sarasvatī river as its western border.
Question Type: factual
Question: What does Macchā in Pāli and Matsya in Sanskrit mean?
Answer: fish

Example 2 (Factoid):
Context: The Vedarthasamgraha is a treatise by the Hindu philosopher Ramanuja, comprising his exegesis of a number of Upanishadic texts. The first of his three major works, Ramanuja propounds the doctrine of the Vishishtadvaita philosophy.
Question Type: factual
Question: Who is the author of the Vedarthasamgraha?
Answer: Ramanuja

Example 3 (Factoid):
Context: Padri-Ki-Haveli, also known as the Visitation of the blessed Virgin Mary, is the oldest church in Bihar. When Roman Catholics arrived in Bihar, they built a small church in 1713 at a place now known as Padri-ki-Haveli. The current church was re-designed by a Venetian architect Tirreto in 1772.
Question Type: factual
Question: When was the small church built in Padri-ki-Haveli?
Answer: 1713

Example 4 (Factoid):
Context: Narayana Teertha composed the Krishna Lila Tarangini, a story of Krishna's life beginning from his birth to his marriage to Rukmini.
Question Type: factual
Question: Who composed the Krishna Lila Tarangini?
Answer: Narayana Teertha

Example 5 (Factoid):
Context: Ashva (Sanskrit: अश्व, romanized: aśva) is the Sanskrit word for a horse, one of the significant animals finding references in the Vedas as well as later Hindu scriptures.
Question Type: factual
Question: What does Ashva mean in Sanskrit?
Answer: Ashva means horse in Sanskrit.
"""

BOOLEAN_EXAMPLES = """
Example 1 (Boolean):
Context: La Martiniere Calcutta is a private school established in 1836 by Claude Martin. It follows the ICSE curriculum and is managed by a private trust.
Question Type: boolean
Question: Is La Martiniere Calcutta a government school?
Answer: No

Example 2 (Boolean):
Context: The Vānaprastha stage of life involves cleansing rituals, meditation, and a gradual withdrawal from worldly duties. Individuals in this stage perform specific rites for purification.
Question Type: boolean
Question: Does Vānaprastha stage of life involve cleansing rituals?
Answer: Yes

Example 3 (Boolean):
Context: The Narmada river is home to several species of wildlife. However, the river is not home to any species currently classified as endangered by the IUCN.
Question Type: boolean
Question: Is the Narmada river home to any endangered species?
Answer: No

Example 4 (Boolean):
Context: Tapas Sen was a pioneering lighting designer who worked extensively with the Bengali theatre movement. He transformed stage lighting into an art form in Indian theatre.
Question Type: boolean
Question: Did Tapas Sen work with Bengali theatre movement?
Answer: Yes

Example 5 (Boolean):
Context: Mahadaji Shinde launched an invasion of Rajputana in 1787 but was defeated by the combined forces of the Rajput rulers and forced to retreat.
Question Type: boolean
Question: Was Mahadaji Shinde successful in his invasion of Rajputana in 1787?
Answer: No
"""

OTHERS_EXAMPLES = """
Example 1 (Other):
Context: Fort Geldria or Fort Geldaria, located in Pulicat, was the seat of the Dutch Republic's first settlement in India, and the capital of Dutch Coromandel. It was built in 1613 and served as a strategic trading post for spice trade.
Question Type: other
Question: Explain the significance of Fort Geldria in Pulicat.
Answer: Fort Geldria or Fort Geldaria, located in Pulicat, was the seat of the Dutch Republic's first settlement in India, and the capital of Dutch Coromandel.

Example 2 (Other):
Context: Vallabha formulated the philosophy of Śuddhādvaita, in response to the Ādvaita Vedānta of Śaṅkara. Vallabha stated that religious disciplines based on knowledge alone were insufficient for liberation and emphasized devotion (bhakti) as the path to salvation.
Question Type: other
Question: Explain the philosophy of Śuddhādvaita.
Answer: Vallabha formulated the philosophy of Śuddhādvaita, in response to the Ādvaita Vedānta of Śaṅkara, which he called Maryādā Mārga or Path of Limitations. Vallabha stated that religious disciplines based on knowledge alone were insufficient for liberation.

Example 3 (Other):
Context: Jugal Kishore Birla started his business career at an early age, joining his father Baldeodas Birla in Calcutta and soon came to be known as a reputed trader and speculator in opium, silver, spice and other trades. He later diversified into manufacturing and industry.
Question Type: other
Question: Explain Jugal Kishore Birla's role in the business world.
Answer: He started his business career at an early age, joining his father Baldeodas Birla in Calcutta and soon came to be known as reputed trader and speculator in opium, silver, spice and other trades.

Example 4 (Other):
Context: Nathamuni obtained the Naalayira Divya Prabandham by singing the pasurams with dedication under the tamarind tree. After completing the count of 12,000 times, Nammalvar himself came in front of Nathamuni and blessed him with the complete text.
Question Type: other
Question: Explain how Nathamuni obtained the Naalayira Divya Prabandham.
Answer: Nathamuni obtained the Naalayira Divya Prabandham by singing the pasurams with dedication under the tamarind tree. After completing the count of 12,000 times, Nammalvar himself came in front of Nathamuni and blessed him with the complete text.

Example 5 (Other):
Context: Belur Math is notable for its architecture that fuses Hindu, Islamic, Buddhist, and Christian art and motifs as a symbol of unity of all religions. The design was envisioned by Swami Vivekananda. The temple resembles a temple, a mosque, or a church depending on the angle of view.
Question Type: other
Question: What is notable about the architecture of Belur Math?
Answer: It fuses Hindu, Islamic, Buddhist, and Christian art and motifs as a symbol of unity of all religions.
"""


def build_prompt(question, context, question_type, **kwargs):
    """Build a five-shot prompt with 5 examples per category."""
    examples = FACTOID_EXAMPLES + BOOLEAN_EXAMPLES + OTHERS_EXAMPLES
    if context and context.strip():
        return (
            f"{SYSTEM_INSTRUCTION_WITH_CONTEXT}\n\n"
            f"Here are examples showing the expected answer format:\n"
            f"{examples}\n"
            f"Now answer the following:\n\n"
            f"Context:\n{context}\n\n"
            f"Question Type: {question_type}\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
    return (
        f"{SYSTEM_INSTRUCTION_NO_CONTEXT}\n\n"
        f"Here are examples showing the expected answer format:\n"
        f"{examples}\n"
        f"Now answer the following:\n\n"
        f"Question Type: {question_type}\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
