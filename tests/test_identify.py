from __future__ import annotations

from types import SimpleNamespace

from vestigia import build_model_fingerprint, test_model_against_fingerprint
from vestigia.llm import LLMResponse


class FakeClient:
    def __init__(self, model: str, answers: list[str], reasoning: str | None = None) -> None:
        self.config = SimpleNamespace(
            model=model, provider="openai_compatible", temperature=0.1, max_tokens=32
        )
        self._answers = iter(answers)
        self.reasoning = reasoning

    def complete(self, prompt: str, **_: object) -> LLMResponse:
        return LLMResponse(
            content=next(self._answers),
            reasoning_content=self.reasoning,
            model=self.config.model,
            provider="openai_compatible",
            finish_reason="stop",
            usage=None,
            request_id=None,
            raw={},
        )


def parse_number(content: str) -> dict[str, str]:
    return {"value": content}


def test_parsed_fingerprint_contains_no_length_distribution() -> None:
    fingerprint = build_model_fingerprint(
        FakeClient("reference", ["76"] * 50),
        "Pick a favorite number.",
        parse_number,
        feature_kind="parsed",
        field="parsed.value",
        count=50,
        subset_size=20,
        resamples=20,
    )

    assert fingerprint.feature_kind == "parsed"
    assert fingerprint.values == ("76",) * 50
    assert fingerprint.distribution == {"76": 1.0}
    assert fingerprint.length_field is None
    assert fingerprint.length_statistics is None


def test_length_fingerprint_contains_no_parsed_distribution() -> None:
    fingerprint = build_model_fingerprint(
        FakeClient("reference", ["76"] * 50, reasoning="reason" * 10),
        "Pick a favorite number.",
        parse_number,
        feature_kind="length",
        length_field="reasoning_content",
        count=50,
        subset_size=20,
        resamples=20,
    )

    assert fingerprint.feature_kind == "length"
    assert fingerprint.field is None
    assert fingerprint.length_field == "reasoning_content"
    assert fingerprint.length_statistics == {
        "mean": 60.0,
        "standard_deviation": 0.0,
        "min": 60.0,
        "max": 60.0,
    }
    assert fingerprint.distribution == {'{"lower":32,"upper_exclusive":64}': 1.0}


def test_matching_parsed_candidate_is_accepted() -> None:
    fingerprint = build_model_fingerprint(
        FakeClient("reference", ["76"] * 50),
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=50,
        subset_size=20,
        resamples=20,
    )

    result = test_model_against_fingerprint(
        FakeClient("candidate", ["76"] * 20), fingerprint, parse_number, count=20
    )

    assert result.feature_kind == "parsed"
    assert result.matches_reference is True
    assert result.length_statistics is None
