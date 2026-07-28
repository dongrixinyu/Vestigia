"""Build and compare one kind of LLM behavioral fingerprint per probe."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from vestigia.fingerprint import canonical_value, resolve_field
from vestigia.llm import LLMClient, LLMResponse
from vestigia.validation import (
    compare_distributions,
    distribution,
    log_length_values,
    text_length_summary,
    validate_stability,
)

Parser = Callable[[str], Mapping[str, Any]]
_STABILITY_SUBSET_SIZE = 20
_STABILITY_RESAMPLES = 1_000
_STABILITY_SEED = 0
_STABILITY_MAX_P95_TV_DISTANCE = 0.20


FeatureKind = Literal["parsed", "length"]
LengthField = Literal["content", "reasoning_content"]


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """One empirical feature distribution for one fixed model request.

    ``feature_kind='parsed'`` stores parser-derived categorical values.
    ``feature_kind='length'`` stores power-of-two character-length buckets from
    one LiteLLM response field. A fingerprint never contains both kinds.
    """

    model: str
    prompt: str
    system: str | None
    temperature: float | None
    max_tokens: int | None
    request_configuration: Mapping[str, Any]
    feature_kind: FeatureKind
    field: str | None
    length_field: LengthField | None
    values: tuple[str, ...]
    distribution: Mapping[str, float]
    stability: Mapping[str, Any]
    length_statistics: Mapping[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FingerprintTestResult:
    """Comparison result for the single feature distribution in a fingerprint."""

    reference_model: str
    tested_model: str
    feature_kind: FeatureKind
    field: str | None
    length_field: LengthField | None
    successful_sample_count: int
    distribution: Mapping[str, float]
    distances: Mapping[str, float]
    acceptance_tv_distance: float
    reference_reliable: bool
    matches_reference: bool
    length_statistics: Mapping[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model_fingerprint(
    client: LLMClient,
    prompt: str,
    parser: Parser,
    *,
    feature_kind: FeatureKind = "parsed",
    field: str = "parsed",
    length_field: LengthField = "content",
    count: int = 50,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ModelFingerprint:
    """Call a model repeatedly and build exactly one selected feature distribution."""
    _validate_count(count, "count")
    _validate_feature_kind(feature_kind)
    request_configuration = _request_configuration(
        client, temperature=temperature, max_tokens=max_tokens)
    values, raw_lengths = _collect_feature_values(
        client,
        prompt,
        parser,
        feature_kind=feature_kind,
        field=field,
        length_field=length_field,
        count=count,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    stability = validate_stability(
        values,
        sample_size=min(_STABILITY_SUBSET_SIZE, len(values)),
        resamples=_STABILITY_RESAMPLES,
        seed=_STABILITY_SEED,
        max_p95_tv_distance=_STABILITY_MAX_P95_TV_DISTANCE,
    )
    return ModelFingerprint(
        model=client.config.model,
        prompt=prompt,
        system=system,
        temperature=temperature if temperature is not None else client.config.temperature,
        max_tokens=max_tokens if max_tokens is not None else client.config.max_tokens,
        request_configuration=request_configuration,
        feature_kind=feature_kind,
        field=field if feature_kind == "parsed" else None,
        length_field=length_field if feature_kind == "length" else None,
        values=tuple(values),
        distribution=distribution(values),
        stability=stability,
        length_statistics=text_length_summary(raw_lengths) if raw_lengths is not None else None,
    )


def test_model_against_fingerprint(
    client: LLMClient,
    fingerprint: ModelFingerprint,
    parser: Parser,
    *,
    count: int = 20,
) -> FingerprintTestResult:
    """Sample and compare only the feature kind stored in ``fingerprint``."""
    _validate_count(count, "count")
    _validate_feature_kind(fingerprint.feature_kind)
    candidate_configuration = _request_configuration(
        client, temperature=fingerprint.temperature, max_tokens=fingerprint.max_tokens
    )
    sampling_keys = (
        "extra_body", "top_p", "top_k", "presence_penalty", "frequency_penalty",
        "reasoning", "reasoning_effort",
    )
    if any(candidate_configuration[key] != fingerprint.request_configuration.get(key) for key in sampling_keys):
        raise ValueError("candidate sampling parameters must match the fingerprint")
    values, raw_lengths = _collect_feature_values(
        client,
        fingerprint.prompt,
        parser,
        feature_kind=fingerprint.feature_kind,
        field=fingerprint.field or "parsed",
        length_field=fingerprint.length_field or "content",
        count=count,
        system=fingerprint.system,
        temperature=fingerprint.temperature,
        max_tokens=fingerprint.max_tokens,
    )
    distances = compare_distributions(fingerprint.values, values)
    acceptance = float(fingerprint.stability["total_variation_distance"]["p95"])
    reliable = bool(fingerprint.stability["reliable"])
    return FingerprintTestResult(
        reference_model=fingerprint.model,
        tested_model=client.config.model,
        feature_kind=fingerprint.feature_kind,
        field=fingerprint.field,
        length_field=fingerprint.length_field,
        successful_sample_count=len(values),
        distribution=distribution(values),
        distances=distances,
        acceptance_tv_distance=acceptance,
        reference_reliable=reliable,
        matches_reference=reliable and distances["total_variation_distance"] <= acceptance,
        length_statistics=text_length_summary(raw_lengths) if raw_lengths is not None else None,
    )


test_model_against_fingerprint.__test__ = False


def _request_configuration(
    client: LLMClient, *, temperature: float | None, max_tokens: int | None
) -> dict[str, Any]:
    return {
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
        "api_version": getattr(client.config, "api_version", None) if client.config.provider == "anthropic" else None,
        "disable_response_cache": getattr(client.config, "disable_response_cache", True),
        "cache_bust_query_param": getattr(client.config, "cache_bust_query_param", None),
    }


def _collect_feature_values(
    client: LLMClient,
    prompt: str,
    parser: Parser,
    *,
    feature_kind: FeatureKind,
    field: str,
    length_field: LengthField,
    count: int,
    system: str | None,
    temperature: float | None,
    max_tokens: int | None,
) -> tuple[list[str], list[int] | None]:
    values: list[str] = []
    raw_lengths: list[int] | None = [] if feature_kind == "length" else None
    for _ in range(count):
        response: LLMResponse = client.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        if feature_kind == "parsed":
            parsed = parser(response.content)
            values.append(_distribution_value(resolve_field({"parsed": parsed}, field)))
        else:
            output = response.content if length_field == "content" else response.reasoning_content or ""
            raw_lengths.append(len(output))  # type: ignore[union-attr]
            values.append(log_length_values([len(output)])[0])
    return values, raw_lengths


def _distribution_value(value: Any) -> str:
    return value if isinstance(value, str) else canonical_value(value)


def _validate_feature_kind(value: str) -> None:
    if value not in {"parsed", "length"}:
        raise ValueError("feature_kind must be 'parsed' or 'length'")


def _validate_count(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
