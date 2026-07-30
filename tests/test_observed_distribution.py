from __future__ import annotations

import json

import pytest

from vestigia.config import SYSTEM_PROMPT
from vestigia.identify import ModelFingerprint
from vestigia.workflow import predict_distribution, save_fingerprint


def _fingerprint(model: str, prompt: str, values: tuple[str, ...]) -> ModelFingerprint:
    return ModelFingerprint(
        model=model,
        prompt=prompt,
        request_configuration={
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 1.0,
            "max_tokens": None,
            "top_p": 1.0,
            "top_k": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "extra_body": {},
            "extra_headers": {},
        },
        feature_kind="parsed",
        field="parsed.first_number.value",
        values=values,
        distribution={},
        stability={},
    )


def _save(tmp_path, model: str, prompt_id: str, prompt: str, values: tuple[str, ...]) -> str:
    path = save_fingerprint(_fingerprint(model, prompt, values), tmp_path, prompt_id=prompt_id)
    return json.loads(path.read_text("utf-8"))["parameters_hash"]


def test_distribution_prediction_aggregates_matching_features_per_model(tmp_path) -> None:
    favorite_hash = _save(tmp_path, "model-a", "favorite_number", "favorite prompt", ("142",) * 10)
    _save(tmp_path, "model-b", "favorite_number", "favorite prompt", ("198",) * 10)
    identity_hash = _save(tmp_path, "model-a", "model_identity", "identity prompt", ("gpt",) * 10)
    _save(tmp_path, "model-b", "model_identity", "identity prompt", ("claude",) * 10)

    result = predict_distribution(
        [
            {"prompt_id": "favorite_number", "params_hash": favorite_hash, "values": ["142"] * 8 + ["198"] * 2},
            {"prompt_id": "model_identity", "params_hash": identity_hash, "values": ["gpt"] * 9 + ["claude"]},
        ],
        tmp_path,
        distance_type="total_variation",
        softmax_temperature=0.1,
    )

    assert result.matches[0].model == "model-a"
    assert result.matches[0].distance == pytest.approx(0.15)
    assert len(result.matches[0].feature_matches) == 2
    assert {item.prompt_id for item in result.matches[0].feature_matches} == {
        "favorite_number", "model_identity"
    }
    assert sum(match.probability for match in result.matches) == pytest.approx(1.0)


def test_distribution_prediction_recursively_loads_vendor_model_fingerprints(tmp_path) -> None:
    fingerprint_directory = tmp_path / "vendor-a" / "model-a"
    fingerprint_hash = _save(
        fingerprint_directory, "model-a", "favorite_number", "favorite prompt", ("142",) * 10
    )

    result = predict_distribution(
        [{"prompt_id": "favorite_number", "params_hash": fingerprint_hash, "values": ["142"]}],
        tmp_path,
    )

    assert result.matches[0].model == "model-a"
    assert result.matches[0].feature_matches[0].fingerprint_path.parent == fingerprint_directory


def test_distribution_prediction_uses_any_params_hash_when_empty(tmp_path) -> None:
    _save(tmp_path, "model-a", "favorite_number", "cold prompt", ("142",) * 10)
    warm_hash = _save(tmp_path, "model-a", "favorite_number", "warm prompt", ("198",) * 10)

    result = predict_distribution(
        [{"prompt_id": "favorite_number", "params_hash": "", "values": ["198"]}],
        tmp_path,
    )

    feature_match = result.matches[0].feature_matches[0]
    assert result.matches[0].model == "model-a"
    assert feature_match.params_hash == warm_hash
    assert feature_match.distance == 0.0
    assert feature_match.distance_type == "total_variation"


def test_distribution_prediction_requires_matching_prompt_and_params_hash(tmp_path) -> None:
    _save(tmp_path, "model-a", "favorite_number", "favorite prompt", ("142",) * 10)

    with pytest.raises(ValueError, match="no saved fingerprints match"):
        predict_distribution(
            [{"prompt_id": "favorite_number", "params_hash": "wrong-hash", "values": ["142"]}],
            tmp_path,
        )


def test_distribution_prediction_rejects_missing_experiment_identity(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing fields"):
        predict_distribution([{"values": ["142"]}], tmp_path)


def test_distribution_prediction_rejects_unknown_distance_type(tmp_path) -> None:
    fingerprint_hash = _save(tmp_path, "model-a", "favorite_number", "favorite prompt", ("142",) * 10)

    with pytest.raises(ValueError, match="unsupported distance_type"):
        predict_distribution(
            [{"prompt_id": "favorite_number", "params_hash": fingerprint_hash, "values": ["142"]}],
            tmp_path,
            distance_type="euclidean",  # type: ignore[arg-type]
        )
