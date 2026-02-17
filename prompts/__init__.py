"""
prompts package - Maps strategy names to prompt builder functions.
"""

from prompts.zero_shot import build_prompt as zero_shot_prompt
from prompts.one_shot import build_prompt as one_shot_prompt
from prompts.three_shot import build_prompt as three_shot_prompt
from prompts.five_shot import build_prompt as five_shot_prompt
from prompts.rag_prompt import build_prompt as rag_prompt

STRATEGY_BUILDERS = {
    "zero_shot": zero_shot_prompt,
    "one_shot": one_shot_prompt,
    "three_shot": three_shot_prompt,
    "five_shot": five_shot_prompt,
    "rag": rag_prompt,
}
