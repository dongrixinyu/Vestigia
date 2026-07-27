from __future__ import annotations

from vestigia.prompts import DEFAULT_PROMPTS, iter_prompts


def test_prompt_sequence_cycles_templates_then_variants() -> None:
    prompts = list(iter_prompts(len(DEFAULT_PROMPTS) + 1))

    assert prompts[0] == (DEFAULT_PROMPTS[0].variants[0], DEFAULT_PROMPTS[0])
    assert prompts[len(DEFAULT_PROMPTS)] == (DEFAULT_PROMPTS[0].variants[1], DEFAULT_PROMPTS[0])
