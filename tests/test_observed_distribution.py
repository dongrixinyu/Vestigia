from __future__ import annotations

import pytest

from vestigia.config import SYSTEM_PROMPT
from vestigia.identify import ModelFingerprint
from vestigia.workflow import predict_distribution, save_fingerprint


def _fingerprint(model: str, values: tuple[str, ...]) -> ModelFingerprint:
    return ModelFingerprint(
        model=model,
        prompt="favorite number",
        request_configuration={
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 1.0,
            "max_tokens": None,
            "top_p": 1.0,
            "top_k": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "reasoning": None,
            "reasoning_effort": None,
            "extra_body": {},
            "extra_headers": {},
        },
        feature_kind="parsed",
        field="parsed.first_number.value",
        values=values,
        distribution={},
        stability={},
    )


def test_distribution_prediction_supports_both_distance_types(tmp_path) -> None:
    save_fingerprint(_fingerprint("model-a", ("142",) * 10), tmp_path, prompt_id="favorite_number")
    save_fingerprint(_fingerprint("model-b", ("198",) * 10), tmp_path, prompt_id="favorite_number")

    tv_result = predict_distribution(
        ["142"] * 8 + ["198"] * 2,
        tmp_path,
        distance_type="total_variation",
        softmax_temperature=0.1,
    )
    js_result = predict_distribution(
        ["142"] * 8 + ["198"] * 2,
        tmp_path,
        distance_type="jensen_shannon",
        softmax_temperature=0.1,
    )

    assert tv_result.distance_type == "total_variation"
    assert tv_result.matches[0].model == "model-a"
    assert tv_result.matches[0].total_variation_distance == pytest.approx(0.2)
    assert tv_result.matches[0].jensen_shannon_distance > 0
    assert sum(match.probability for match in tv_result.matches) == pytest.approx(1.0)

    assert js_result.distance_type == "jensen_shannon"
    assert js_result.matches[0].model == "model-a"
    assert js_result.matches[0].jensen_shannon_distance > 0
    assert sum(match.probability for match in js_result.matches) == pytest.approx(1.0)


def test_distribution_prediction_rejects_unknown_distance_type(tmp_path) -> None:
    save_fingerprint(_fingerprint("model-a", ("142",) * 10), tmp_path, prompt_id="favorite_number")

    with pytest.raises(ValueError, match="unsupported distance_type"):
        predict_distribution(["142"], tmp_path, distance_type="euclidean")  # type: ignore[arg-type]
