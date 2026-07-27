"""High-level public API for building and testing LLM behavioral fingerprints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vestigia.fingerprint import canonical_value, resolve_field
from vestigia.llm import LLMClient, LLMResponse
from vestigia.validation import (
    compare_distributions,
    distribution,
    log_length_values,
    text_length_summary,
    total_variation_distance,
    validate_length_distribution,
    validate_stability,
)

Parser = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """Validated categorical and response-length reference features for one model."""

    model: str
    provider: str
    prompt: str
    system: str | None
    temperature: float | None
    max_tokens: int | None
    request_configuration: Mapping[str, Any]
    field: str
    values: tuple[str, ...]
    distribution: Mapping[str, float]
    text_length: Mapping[str, Any]
    stability: Mapping[str, Any]
    length_source: str = "text"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the fingerprint."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FingerprintTestResult:
    """Result of testing repeated candidate outputs against a reference."""

    reference_model: str
    tested_model: str
    field: str
    successful_sample_count: int
    distribution: Mapping[str, float]
    text_length: Mapping[str, Any]
    distances: Mapping[str, float]
    acceptance_tv_distance: float
    reference_reliable: bool
    matches_reference: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable test report."""
        return asdict(self)


def build_model_fingerprint(
    client: LLMClient,
    prompt: str,
    parser: Parser,
    *,
    count: int = 50,
    field: str = "parsed",
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    subset_size: int = 20,
    resamples: int = 1_000,
    seed: int | None = 0,
    max_p95_tv_distance: float = 0.20,
    max_p95_length_tv_distance: float = 0.20,
    length_source: str = "text",
) -> ModelFingerprint:
    """Call a model repeatedly and build distribution and text-length features.

    Text length is raw Unicode character count of ``LLMResponse.text``;
    whitespace and punctuation are included so the metric stays deterministic.
    """
    _validate_count(count, "count")
    _validate_length_source(length_source)
    request_configuration = _request_configuration(
        client, temperature=temperature, max_tokens=max_tokens
    )
    values, text_lengths = _collect_values(
        client,
        prompt,
        parser,
        count=count,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        field=field,
        length_source=length_source,
    )
    stability = validate_stability(
        values,
        sample_size=subset_size,
        resamples=resamples,
        seed=seed,
        max_p95_tv_distance=max_p95_tv_distance,
    )
    text_length = {
        **validate_length_distribution(
            text_lengths,
            sample_size=subset_size,
            resamples=resamples,
            seed=seed,
            max_p95_tv_distance=max_p95_length_tv_distance,
        ),
        "source": length_source,
    }
    return ModelFingerprint(
        model=client.config.model,
        provider=client.config.provider,
        prompt=prompt,
        system=system,
        temperature=temperature if temperature is not None else client.config.temperature,
        max_tokens=max_tokens if max_tokens is not None else client.config.max_tokens,
        request_configuration=request_configuration,
        field=field,
        values=tuple(values),
        distribution=distribution(values),
        text_length=text_length,
        stability=stability,
        length_source=length_source,
    )


def test_model_against_fingerprint(
    client: LLMClient,
    fingerprint: ModelFingerprint,
    parser: Parser,
    *,
    count: int = 20,
) -> FingerprintTestResult:
    """Call a candidate and require both response distribution and length to match."""
    _validate_count(count, "count")
    candidate_configuration = _request_configuration(
        client, temperature=fingerprint.temperature, max_tokens=fingerprint.max_tokens
    )
    sampling_keys = (
        "extra_body",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "reasoning",
        "reasoning_effort",
    )
    if any(
        candidate_configuration[key] != fingerprint.request_configuration.get(key)
        for key in sampling_keys
    ):
        raise ValueError("candidate sampling parameters must match the fingerprint")
    values, text_lengths = _collect_values(
        client,
        fingerprint.prompt,
        parser,
        count=count,
        system=fingerprint.system,
        temperature=fingerprint.temperature,
        max_tokens=fingerprint.max_tokens,
        field=fingerprint.field,
        length_source=fingerprint.length_source,
    )
    distances = compare_distributions(fingerprint.values, values)
    acceptance = float(fingerprint.stability["total_variation_distance"]["p95"])
    reference_reliable = bool(fingerprint.stability["reliable"])
    reference_length_reliable = bool(fingerprint.text_length["stability"]["reliable"])
    candidate_length = text_length_summary(text_lengths)
    candidate_length_buckets = distribution(log_length_values(text_lengths))
    reference_length_buckets = fingerprint.text_length["distribution"]
    length_distances = {
        "total_variation_distance": total_variation_distance(
            reference_length_buckets, candidate_length_buckets
        )
    }
    length_acceptance = float(
        fingerprint.text_length["stability"]["total_variation_distance"]["p95"]
    )
    length_result = {
        **candidate_length,
        "source": fingerprint.length_source,
        "bucket_scheme": fingerprint.text_length["bucket_scheme"],
        "distribution": candidate_length_buckets,
        "distances": length_distances,
        "acceptance_tv_distance": length_acceptance,
        "reference_reliable": reference_length_reliable,
        "matches_reference": (
            reference_length_reliable
            and length_distances["total_variation_distance"] <= length_acceptance
        ),
    }
    return FingerprintTestResult(
        reference_model=fingerprint.model,
        tested_model=client.config.model,
        field=fingerprint.field,
        successful_sample_count=len(values),
        distribution=distribution(values),
        text_length=length_result,
        distances=distances,
        acceptance_tv_distance=acceptance,
        reference_reliable=reference_reliable,
        matches_reference=(
            reference_reliable
            and reference_length_reliable
            and distances["total_variation_distance"] <= acceptance
            and length_distances["total_variation_distance"] <= length_acceptance
        ),
    )


# Prevent pytest from mistaking this public API function for a test when imported.
test_model_against_fingerprint.__test__ = False


def _request_configuration(
    client: LLMClient, *, temperature: float | None, max_tokens: int | None
) -> dict[str, Any]:
    """Return non-secret controls that define a comparable sampling configuration."""
    return {
        "provider": client.config.provider,
        "model": client.config.model,
        "temperature": temperature if temperature is not None else client.config.temperature,
        "max_tokens": max_tokens if max_tokens is not None else client.config.max_tokens,
        "extra_body": dict(getattr(client.config, "extra_body", {})),
        "top_p": getattr(client.config, "top_p", None),
        "top_k": getattr(client.config, "top_k", None),
        "presence_penalty": getattr(client.config, "presence_penalty", None),
        "frequency_penalty": getattr(client.config, "frequency_penalty", None),
        "reasoning": getattr(client.config, "reasoning", None),
        "reasoning_effort": getattr(client.config, "reasoning_effort", None),
        "api_version": (
            getattr(client.config, "api_version", None)
            if client.config.provider == "anthropic"
            else None
        ),
        "disable_response_cache": getattr(client.config, "disable_response_cache", True),
        "cache_bust_query_param": getattr(client.config, "cache_bust_query_param", None),
    }


def _collect_values(
    client: LLMClient,
    prompt: str,
    parser: Parser,
    *,
    count: int,
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
    field: str,
    length_source: str,
) -> tuple[list[str], list[int]]:
    values: list[str] = []
    text_lengths: list[int] = []
    for _ in range(count):
        response: LLMResponse = client.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        parsed = parser(response.text)
        values.append(canonical_value(resolve_field({"parsed": parsed}, field)))
        text_lengths.append(len(_length_text(response, length_source)))
    return values, text_lengths


def _length_text(response: LLMResponse, source: str) -> str:
    """Return the configured response component used for length features."""
    if source == "text":
        return response.text
    if source == "reasoning_content":
        return response.reasoning_content or ""
    raise ValueError(f"unsupported length source: {source}")


def _validate_length_source(value: str) -> None:
    if value not in {"text", "reasoning_content"}:
        raise ValueError("length_source must be 'text' or 'reasoning_content'")


def _validate_count(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
