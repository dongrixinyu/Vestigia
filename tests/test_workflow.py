from __future__ import annotations

from vestigia.identify import ModelFingerprint
from vestigia.prompts.favorite_number import PROMPT as FAVORITE_NUMBER
from vestigia.workflow import _select_prompt, load_fingerprint, save_fingerprint


def test_saved_fingerprint_can_be_loaded_without_http_or_mock_libraries(tmp_path) -> None:
    fingerprint = ModelFingerprint(
        model="reference-model",
        provider="openai_compatible",
        prompt="Reply with one word.",
        system=None,
        temperature=0.1,
        max_tokens=16,
        request_configuration={"extra_body": {"top_p": 0.9}},
        feature_kind="parsed",
        field="parsed.text",
        length_field=None,
        values=("yes",),
        distribution={"yes": 1.0},
        stability={"reliable": True},
    )
    output = tmp_path / "reference-fingerprint.json"

    save_fingerprint(fingerprint, output)
    assert load_fingerprint(output) == fingerprint


def test_selecting_a_builtin_prompt_uses_its_fixed_variant_and_parser() -> None:
    prompt, parser, feature_kind, length_field = _select_prompt(
        prompt_id="favorite_number", variant_index=0, parser=None
    )

    assert prompt == FAVORITE_NUMBER.variants[0]
    assert parser is FAVORITE_NUMBER.parser
    assert feature_kind == "parsed"
    assert length_field == "content"
