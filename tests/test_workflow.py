from __future__ import annotations

import json

import pytest

from vestigia.config import SYSTEM_PROMPT
from vestigia.identify import ModelFingerprint
from vestigia.prompts.favorite_number import PROMPT as FAVORITE_NUMBER
from vestigia.prompts.project_success_score import PROMPT as PROJECT_SUCCESS_SCORE
from vestigia.workflow import _select_prompt, load_fingerprint, save_fingerprint


def test_saved_fingerprint_can_be_loaded_without_http_or_mock_libraries(tmp_path) -> None:
    fingerprint = ModelFingerprint(
        model="reference-model",
        prompt="Reply with one word.",
        request_configuration={
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 0.1,
            "max_tokens": 16,
            "top_p": 1.0,
            "top_k": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "reasoning": None,
            "reasoning_effort": None,
            "extra_body": {"top_p": 0.9},
            "extra_headers": {},
        },
        feature_kind="parsed",
        field="parsed.text",
        values=("yes",),
        distribution={"yes": 1.0},
        stability={"reliable": True},
    )
    output_directory = tmp_path / "fingerprints"

    output = save_fingerprint(fingerprint, output_directory, prompt_id="custom_probe")

    assert output.parent == output_directory
    assert output.name.startswith("reference-model__custom_probe__")
    assert output.suffix == ".json"
    payload = json.loads(output.read_text("utf-8"))
    assert payload["prompt_id"] == "custom_probe"
    assert len(payload["parameters_hash"]) == 16
    assert payload["request_params"] == {
        "system_prompt": SYSTEM_PROMPT,
        "temperature": 0.1,
        "max_tokens": 16,
        "top_p": 1.0,
        "top_k": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "reasoning": None,
        "reasoning_effort": None,
        "extra_body": {"top_p": 0.9},
        "extra_headers": {},
    }
    assert "request_configuration" not in payload
    assert "system" not in payload
    assert "length_field" not in payload
    assert "length_statistics" not in payload
    assert "effective_request_params" not in payload
    loaded = load_fingerprint(output)
    assert loaded.model == fingerprint.model
    assert loaded.prompt == fingerprint.prompt
    assert loaded.values == fingerprint.values
    assert loaded.request_configuration["extra_body"] == {"top_p": 0.9}


def test_save_fingerprint_uses_parameters_to_separate_files(tmp_path) -> None:
    common = dict(
        model="reference-model", prompt=FAVORITE_NUMBER.variants[0],
        feature_kind="parsed", field="parsed.text",
        values=("yes",), distribution={"yes": 1.0}, stability={"reliable": True},
    )
    cold = ModelFingerprint(request_configuration={"system_prompt": SYSTEM_PROMPT, "temperature": 0.1}, **common)
    warm = ModelFingerprint(request_configuration={"system_prompt": SYSTEM_PROMPT, "temperature": 0.7}, **common)

    cold_path = save_fingerprint(cold, tmp_path, prompt_id="favorite_number")
    warm_path = save_fingerprint(warm, tmp_path, prompt_id="favorite_number")

    assert cold_path != warm_path


def test_save_fingerprint_requires_a_prompt_id_for_unknown_prompts(tmp_path) -> None:
    fingerprint = ModelFingerprint(
        model="reference-model", prompt="unknown", request_configuration={}, feature_kind="parsed",
        field="parsed.text", values=("yes",), distribution={"yes": 1.0},
        stability={"reliable": True},
    )

    with pytest.raises(ValueError, match="prompt_id"):
        save_fingerprint(fingerprint, tmp_path)


def test_selecting_a_builtin_prompt_uses_its_fixed_variant_and_parser() -> None:
    prompt, parser, field, feature_kind, length_field = _select_prompt(
        prompt_id="favorite_number", variant_index=0
    )

    assert prompt == FAVORITE_NUMBER.variants[0]
    assert parser is FAVORITE_NUMBER.parser
    assert field == "parsed.first_number.value"
    assert feature_kind == "parsed"
    assert length_field == "content"


def test_builtin_prompts_own_their_fingerprint_fields() -> None:
    assert FAVORITE_NUMBER.field == "parsed.first_number.value"
    assert PROJECT_SUCCESS_SCORE.field == "parsed.score.value"
