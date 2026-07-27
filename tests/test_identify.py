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
            content=next(self._answers),
            model=self.config.model,
            provider="openai_compatible",
            finish_reason="stop",
            usage=None,
            request_id=None,
            raw={},
        )


def parse_number(content: str) -> dict[str, str]:
    return {"value": content}


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
    assert fingerprint.text_length["statistics"]["mean"] == 2.0
    assert fingerprint.distribution == {"76": 1.0}
    assert result.matches_reference is True


def test_string_feature_values_are_not_json_quoted() -> None:
    fingerprint = build_model_fingerprint(
        FakeClient("reference", ["137"] * 2),
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=2,
        subset_size=1,
        resamples=1,
    )

    assert fingerprint.values == ("137", "137")
    assert fingerprint.distribution == {"137": 1.0}


def test_reasoning_content_can_be_the_length_field() -> None:
    class ReasoningClient(FakeClient):
        def complete(self, prompt: str, **kwargs: object) -> LLMResponse:
            self.calls.append({"prompt": prompt, **kwargs})
            return LLMResponse(
                content=next(self._answers),
                reasoning_content="reason" * 10,
                model=self.config.model,
                provider="openai_compatible",
                finish_reason="stop",
                usage=None,
                request_id=None,
                raw={},
            )

    fingerprint = build_model_fingerprint(
        ReasoningClient("reference", ["101"] * 50),
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=50,
        subset_size=20,
        resamples=20,
        length_field="reasoning_content",
    )

    assert fingerprint.length_field == "reasoning_content"
    assert fingerprint.text_length["field"] == "reasoning_content"
    assert fingerprint.text_length["statistics"]["mean"] == 60.0
