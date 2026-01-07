"""Prompt templates for the plan_generator module."""

from .algorithm_analysis import PAPER_ALGORITHM_ANALYSIS_PROMPT
from .concept_analysis import PAPER_CONCEPT_ANALYSIS_PROMPT
from .code_planning import CODE_PLANNING_PROMPT
from .code_generation import CODE_GENERATION_PROMPT, get_code_generation_prompt

__all__ = [
    "PAPER_ALGORITHM_ANALYSIS_PROMPT",
    "PAPER_CONCEPT_ANALYSIS_PROMPT",
    "CODE_PLANNING_PROMPT",
    "CODE_GENERATION_PROMPT",
    "get_code_generation_prompt",
]
