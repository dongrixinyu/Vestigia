from __future__ import annotations

import pytest

from vestigia.config import SYSTEM_PROMPT
from vestigia.identify import ModelFingerprint
from vestigia.workflow import identify_observed_distribution, save_fingerprint


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


def test_observed_distribution_is_ranked_with_softmax_probabilities(tmp_path) -> None:
    save_fingerprint(_fingerprint("model-a", ("142",) * 10), tmp_path, prompt_id="favorite_number")
    save_fingerprint(_fingerprint("model-b", ("198",) * 10), tmp_path, prompt_id="favorite_number")

    result = identify_observed_distribution(
        ["142"] * 8 + ["198"] * 2, tmp_path, softmax_temperature=0.1
    )

    assert result.matches[0].model == "model-a"
    assert result.matches[0].total_variation_distance == pytest.approx(0.2)
    assert sum(match.probability for match in result.matches) == pytest.approx(1.0)
