from __future__ import annotations

from types import SimpleNamespace

from vestigia import build_model_fingerprint, test_model_against_fingerprint
from vestigia.llm import LLMResponse


class FakeClient:
    def __init__(self, model: str, answers: list[str]) -> None:
        self.config = SimpleNamespace(
            model=model,
            provider="openai_compatible",
            temperature=0.1,
            max_tokens=32,
        )
        self._answers = iter(answers)
        self.calls: list[dict[str, object]] = []

    def complete(self, prompt: str, **kwargs: object) -> LLMResponse:
        self.calls.append({"prompt": prompt, **kwargs})
        return LLMResponse(
            text=next(self._answers),
            model=self.config.model,
            provider="openai_compatible",
            finish_reason="stop",
            usage=None,
            request_id=None,
            raw={},
        )


def parse_number(text: str) -> dict[str, str]:
    return {"value": text}


def test_public_api_builds_reference_then_accepts_matching_candidate() -> None:
    reference_client = FakeClient("claude-reference", ["76"] * 50)
    fingerprint = build_model_fingerprint(
        reference_client,
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=50,
        subset_size=20,
        resamples=20,
        seed=1,
    )
    candidate_client = FakeClient("unknown", ["76"] * 20)

    result = test_model_against_fingerprint(candidate_client, fingerprint, parse_number, count=20)

    assert fingerprint.stability["reliable"] is True
    assert fingerprint.distribution == {'"76"': 1.0}
    assert result.matches_reference is True
    assert result.distances["total_variation_distance"] == 0.0
    assert candidate_client.calls[0] == {
        "prompt": "Pick a favorite number.",
        "system": None,
        "temperature": 0.1,
        "max_tokens": 32,
    }


def test_public_api_rejects_a_different_output_distribution() -> None:
    fingerprint = build_model_fingerprint(
        FakeClient("claude-reference", ["76"] * 50),
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=50,
        subset_size=20,
        resamples=20,
    )

    result = test_model_against_fingerprint(
        FakeClient("different-model", ["36"] * 20), fingerprint, parse_number
    )

    assert result.reference_reliable is True
    assert result.distances["total_variation_distance"] == 1.0
    assert result.matches_reference is False
