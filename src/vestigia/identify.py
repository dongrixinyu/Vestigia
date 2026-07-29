"""Build and compare one kind of LLM behavioral fingerprint per probe."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from vestigia.config import (
    MAX_P95_TV_DISTANCE,
    STABILITY_RESAMPLES,
    STABILITY_SEED,
    STABILITY_SUBSET_SIZE,
    SYSTEM_PROMPT,
)
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
    request_configuration: Mapping[str, Any]
    feature_kind: FeatureKind
    field: str | None
    values: tuple[str, ...]
    distribution: Mapping[str, float]
    stability: Mapping[str, Any]
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FingerprintTestResult:
    """Comparison result for the single feature distribution in a fingerprint."""

    reference_model: str
    tested_model: str
    feature_kind: FeatureKind
    field: str | None
    successful_sample_count: int
    distribution: Mapping[str, float]
    distances: Mapping[str, float]
    acceptance_tv_distance: float
    reference_reliable: bool
    matches_reference: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FingerprintIdentificationResult:
    """Candidate fingerprint compared against every compatible historical reference."""

    tested_model: str
    comparisons: tuple[FingerprintTestResult, ...]
    best_match: FingerprintTestResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tested_model": self.tested_model,
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
            "best_match": self.best_match.to_dict() if self.best_match is not None else None,
        }


def build_model_fingerprint(
    client: LLMClient,
    prompt: str,
    parser: Parser,
    *,
    feature_kind: FeatureKind = "parsed",
    field: str = "parsed",
    length_field: LengthField = "content",
    count: int = 50,
) -> ModelFingerprint:
    """Call a model repeatedly and build exactly one selected feature distribution."""
    _validate_count(count, "count")
    _validate_feature_kind(feature_kind)
    request_configuration = _request_configuration(client)
    started_at = _utc_timestamp()
    values, _ = _collect_feature_values(
        client,
        prompt,
        parser,
        feature_kind=feature_kind,
        field=field,
        length_field=length_field,
        count=count,
    )
    finished_at = _utc_timestamp()
    stability = validate_stability(
        values,
        sample_size=min(STABILITY_SUBSET_SIZE, len(values)),
        resamples=STABILITY_RESAMPLES,
        seed=STABILITY_SEED,
        max_p95_tv_distance=MAX_P95_TV_DISTANCE,
    )
    return ModelFingerprint(
        model=client.config.model,
        prompt=prompt,
        request_configuration=request_configuration,
        feature_kind=feature_kind,
        field=field if feature_kind == "parsed" else None,
        values=tuple(values),
        distribution=distribution(values),
        stability=stability,
        started_at=started_at,
        finished_at=finished_at,
    )


def _effective_system_prompt(system: str | None) -> str:
    """Return the mandatory system instruction actually sent by ``LLMClient``."""
    if system is None or system == SYSTEM_PROMPT:
        return SYSTEM_PROMPT
    if system.startswith(f"{SYSTEM_PROMPT}\n\n"):
        return system
    return f"{SYSTEM_PROMPT}\n\n{system}"


def _utc_timestamp() -> str:
    """Return an unambiguous RFC 3339 timestamp in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def compare_fingerprint_to_reference(
    candidate: ModelFingerprint, reference: ModelFingerprint
) -> FingerprintTestResult:
    """Compare two already-collected fingerprints without making LLM calls."""
    _validate_compatible_fingerprints(candidate, reference)
    distances = compare_distributions(reference.values, candidate.values)
    acceptance = float(reference.stability["total_variation_distance"]["p95"])
    reliable = bool(reference.stability["reliable"])
    return FingerprintTestResult(
        reference_model=reference.model,
        tested_model=candidate.model,
        feature_kind=reference.feature_kind,
        field=reference.field,
        successful_sample_count=len(candidate.values),
        distribution=candidate.distribution,
        distances=distances,
        acceptance_tv_distance=acceptance,
        reference_reliable=reliable,
        matches_reference=reliable and distances["total_variation_distance"] <= acceptance,
    )


def _validate_compatible_fingerprints(
    candidate: ModelFingerprint, reference: ModelFingerprint
) -> None:
    """Reject comparisons made under different experimental conditions."""
    candidate_configuration = dict(candidate.request_configuration)
    reference_configuration = dict(reference.request_configuration)
    candidate_configuration.pop("model", None)
    reference_configuration.pop("model", None)
    if (
        candidate.prompt != reference.prompt
        or candidate.feature_kind != reference.feature_kind
        or candidate.field != reference.field
        or candidate_configuration != reference_configuration
    ):
        raise ValueError(
            "fingerprints must use the same prompt, feature settings, and request parameters"
        )


def _request_configuration(
    client: LLMClient,
) -> dict[str, Any]:
    """Return the complete request parameters recorded with a fingerprint."""
    return {
        "system_prompt": SYSTEM_PROMPT,
        "temperature": client.config.temperature,
        "max_tokens": client.config.max_tokens,
        "top_p": getattr(client.config, "top_p", 1.0),
        "top_k": getattr(client.config, "top_k", None),
        "presence_penalty": getattr(client.config, "presence_penalty", 0.0),
        "frequency_penalty": getattr(client.config, "frequency_penalty", 0.0),
        "extra_body": dict(getattr(client.config, "extra_body", {})),
        "extra_headers": dict(getattr(client.config, "extra_headers", {})),
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
) -> tuple[list[str], list[int] | None]:
    values: list[str] = []
    raw_lengths: list[int] | None = [] if feature_kind == "length" else None
    for _ in range(count):
        response: LLMResponse = client.complete(prompt)
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
