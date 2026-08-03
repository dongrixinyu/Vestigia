from __future__ import annotations

from threading import Lock
from time import sleep
from types import SimpleNamespace

import pytest

import vestigia.identify as identify
from vestigia import build_model_fingerprint, compare_fingerprint_to_reference
from vestigia.llm import LLMRequestError, LLMResponse


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
    )

    assert fingerprint.feature_kind == "parsed"
    assert fingerprint.values == ("76",) * 50
    assert fingerprint.distribution == {"76": 1.0}
    assert fingerprint.request_configuration["system_prompt"]
    assert fingerprint.request_configuration["temperature"] == 0.1


def test_length_fingerprint_contains_no_parsed_distribution() -> None:
    fingerprint = build_model_fingerprint(
        FakeClient("reference", ["76"] * 50, reasoning="reason" * 10),
        "Pick a favorite number.",
        parse_number,
        feature_kind="length",
        length_field="reasoning_content",
        count=50,
    )

    assert fingerprint.feature_kind == "length"
    assert fingerprint.field is None
    assert fingerprint.request_configuration["system_prompt"]
    assert fingerprint.distribution == {'{"lower":32,"upper_exclusive":64}': 1.0}


def test_collected_fingerprints_compare_without_another_model_call() -> None:
    reference = build_model_fingerprint(
        FakeClient("reference", ["76"] * 50),
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=50,
    )
    candidate = build_model_fingerprint(
        FakeClient("unknown", ["76"] * 20),
        "Pick a favorite number.",
        parse_number,
        field="parsed.value",
        count=20,
    )

    result = compare_fingerprint_to_reference(candidate, reference)

    assert result.reference_model == "reference"
    assert result.tested_model == "unknown"
    assert result.matches_reference is True


def test_rate_limit_downgrades_remaining_fingerprint_collection_to_sequential(monkeypatch) -> None:
    class RateLimitedClient:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                model="rate-limited", provider="openai_compatible", temperature=0.1, max_tokens=32
            )
            self.active = 0
            self.max_active = 0
            self.calls = 0
            self.lock = Lock()
            self.rate_limited = False

        def complete(self, _prompt: str, **_: object) -> LLMResponse:
            with self.lock:
                self.calls += 1
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                should_rate_limit = self.active > 1 and not self.rate_limited
                if should_rate_limit:
                    self.rate_limited = True
            sleep(0.01)
            with self.lock:
                self.active -= 1
            if should_rate_limit:
                raise LLMRequestError("Rate limit reached", status_code=429)
            return LLMResponse(
                content="76", reasoning_content=None, model="rate-limited",
                provider="openai_compatible", finish_reason="stop", usage=None,
                request_id=None, raw={},
            )

    monkeypatch.setattr(identify, "LLM_COLLECTION_CONCURRENCY", 2)
    client = RateLimitedClient()

    fingerprint = build_model_fingerprint(
        client, "Pick a number.", parse_number, field="parsed.value", count=4
    )

    assert fingerprint.values == ("76",) * 4
    assert client.max_active == 2  # The first batch started concurrently.
    assert client.calls == 5  # One rate-limited item was retried serially.
