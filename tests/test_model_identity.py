from __future__ import annotations

import pytest

from vestigia.prompts.model_identity import CATEGORIES, NULL, PROMPT, SYSTEM


@pytest.mark.parametrize("response", CATEGORIES)
def test_model_identity_accepts_each_exact_category(response: str) -> None:
    parsed = PROMPT.parser(response)

    assert parsed == {"identity": response}
    assert PROMPT.checker(response, parsed)


@pytest.mark.parametrize(
    "response",
    ["GPT", "gpt\n", "gpt。", "我是 GPT", "gpt 或 claude", "", "unknown", "perf LLM"],
)
def test_model_identity_maps_invalid_response_to_null(response: str) -> None:
    parsed = PROMPT.parser(response)

    assert parsed == {"identity": NULL}
    assert not PROMPT.checker(response, parsed)


def test_model_identity_uses_system_instruction_and_automatic_field() -> None:
    assert PROMPT.system == SYSTEM
    assert PROMPT.field == "parsed.identity"
    assert len(PROMPT.variants) == 5
    assert all("必须且只能" in variant for variant in PROMPT.variants)
