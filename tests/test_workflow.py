from __future__ import annotations

from vestigia.identify import ModelFingerprint
from vestigia.workflow import _select_prompt, load_fingerprint, save_fingerprint
from vestigia.prompts.favorite_number import PROMPT as FAVORITE_NUMBER


def test_saved_fingerprint_can_be_loaded_without_http_or_mock_libraries(tmp_path) -> None:
    fingerprint = ModelFingerprint(
        model="reference-model",
        provider="openai_compatible",
        prompt="Reply with one word.",
        system=None,
        temperature=0.1,
        max_tokens=16,
        request_configuration={"extra_body": {"top_p": 0.9}},
        field="parsed.text",
        values=("yes",),
        distribution={"yes": 1.0},
        text_length={"stability": {"reliable": True}},
        stability={"reliable": True},
        length_field="content",
    )
    output = tmp_path / "reference-fingerprint.json"

    save_fingerprint(fingerprint, output)
    loaded = load_fingerprint(output)
    assert loaded == fingerprint


def test_selecting_a_builtin_prompt_uses_its_fixed_variant_and_parser() -> None:
    prompt, parser, length_field = _select_prompt(
        prompt_id="favorite_number",
        variant_index=0,
        parser=None,
    )

    assert prompt == FAVORITE_NUMBER.variants[0]
    assert parser is FAVORITE_NUMBER.parser
    assert length_field == "reasoning_content"


def test_select_prompt_rejects_an_invalid_variant() -> None:
    import pytest

    with pytest.raises(ValueError, match="out of range"):
        _select_prompt(
            prompt_id="favorite_number",
            variant_index=len(FAVORITE_NUMBER.variants),
            parser=None,
        )
