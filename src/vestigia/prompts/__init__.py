"""Fixed fingerprint probes, each implemented in its own module."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from vestigia.prompts.ambiguous_choice import PROMPT as AMBIGUOUS_CHOICE
from vestigia.prompts.base import Checker, Parser, PromptTemplate
from vestigia.prompts.creative_association import PROMPT as CREATIVE_ASSOCIATION
from vestigia.prompts.everyday_advice import PROMPT as EVERYDAY_ADVICE
from vestigia.prompts.favorite_number import PROMPT as FAVORITE_NUMBER
from vestigia.prompts.instruction_following import PROMPT as INSTRUCTION_FOLLOWING
from vestigia.prompts.project_success_score import PROMPT as PROJECT_SUCCESS_SCORE
from vestigia.prompts.short_self_description import PROMPT as SHORT_SELF_DESCRIPTION
from vestigia.prompts.simple_explanation import PROMPT as SIMPLE_EXPLANATION
from vestigia.prompts.word_association import PROMPT as WORD_ASSOCIATION

DEFAULT_PROMPTS: tuple[PromptTemplate, ...] = (
    FAVORITE_NUMBER,
    PROJECT_SUCCESS_SCORE,
    SHORT_SELF_DESCRIPTION,
    CREATIVE_ASSOCIATION,
    AMBIGUOUS_CHOICE,
    INSTRUCTION_FOLLOWING,
    EVERYDAY_ADVICE,
    WORD_ASSOCIATION,
    SIMPLE_EXPLANATION,
)


def iter_prompts(
    count: int, templates: Sequence[PromptTemplate] = DEFAULT_PROMPTS
) -> Iterator[tuple[str, PromptTemplate]]:
    """Yield exactly ``count`` probes, cycling templates and their wordings evenly."""
    if count < 1:
        raise ValueError("count must be greater than zero")
    if not templates:
        raise ValueError("templates must not be empty")
    for index in range(count):
        template = templates[index % len(templates)]
        variant = template.variants[(index // len(templates)) % len(template.variants)]
        yield variant, template


__all__ = [
    "Checker",
    "DEFAULT_PROMPTS",
    "Parser",
    "PromptTemplate",
    "iter_prompts",
]
