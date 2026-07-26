"""High-level public API for building and testing LLM behavioral fingerprints."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from vestigia.fingerprint import canonical_value, resolve_field
from vestigia.llm import LLMClient, LLMResponse
from vestigia.validation import compare_distributions, distribution, validate_stability

Parser = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class ModelFingerprint:
    """A reference feature distribution obtained from repeated calls to one model.

    ``values`` retain the individual canonical feature values, allowing the
    stability test to be repeated with a different threshold or subset size.
    They may be omitted when serialising a compact, distribution-only report.
    """

    model: str
    provider: str
    prompt: str
    system: str | None
    temperature: float | None
    max_tokens: int | None
    field: str
    values: tuple[str, ...]
    distribution: Mapping[str, float]
    stability: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the fingerprint."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FingerprintTestResult:
    """Result of testing a model's repeated output against a reference."""

    reference_model: str
    tested_model: str
    field: str
    successful_sample_count: int
    distribution: Mapping[str, float]
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
) -> ModelFingerprint:
    """Call a model repeatedly and construct a validated reference distribution.

    ``parser`` receives each final text and must return the feature to count.
    For example, pass ``favorite_number.parse`` and leave ``field`` as
    ``"parsed"`` to count its complete parsed output, or provide a parser that
    returns ``{"value": "76"}`` and use ``field="value"``.
    """
    _validate_count(count, "count")
    values = _collect_values(
        client,
        prompt,
        parser,
        count=count,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        field=field,
    )
    stability = validate_stability(
        values,
        sample_size=subset_size,
        resamples=resamples,
        seed=seed,
        max_p95_tv_distance=max_p95_tv_distance,
    )
    return ModelFingerprint(
        model=client.config.model,
        provider=client.config.provider,
        prompt=prompt,
        system=system,
        temperature=temperature if temperature is not None else client.config.temperature,
        max_tokens=max_tokens if max_tokens is not None else client.config.max_tokens,
        field=field,
        values=tuple(values),
        distribution=distribution(values),
        stability=stability,
    )


def test_model_against_fingerprint(
    client: LLMClient,
    fingerprint: ModelFingerprint,
    parser: Parser,
    *,
    count: int = 20,
) -> FingerprintTestResult:
    """Call a candidate model and determine whether it matches a fingerprint.

    The candidate receives the exact prompt and generation settings used for
    the reference. It matches only if the reference was stable and the
    candidate's TV distance is no greater than the reference subset TV p95.
    This deliberately conservative rule prevents ordinary sampling variation
    from being treated as a model identity difference.
    """
    _validate_count(count, "count")
    values = _collect_values(
        client,
        fingerprint.prompt,
        parser,
        count=count,
        system=fingerprint.system,
        temperature=fingerprint.temperature,
        max_tokens=fingerprint.max_tokens,
        field=fingerprint.field,
    )
    distances = compare_distributions(fingerprint.values, values)
    acceptance = float(fingerprint.stability["total_variation_distance"]["p95"])
    reference_reliable = bool(fingerprint.stability["reliable"])
    return FingerprintTestResult(
        reference_model=fingerprint.model,
        tested_model=client.config.model,
        field=fingerprint.field,
        successful_sample_count=len(values),
        distribution=distribution(values),
        distances=distances,
        acceptance_tv_distance=acceptance,
        reference_reliable=reference_reliable,
        matches_reference=reference_reliable
        and distances["total_variation_distance"] <= acceptance,
    )


# Prevent pytest from mistaking this public API function for a test when imported.
test_model_against_fingerprint.__test__ = False


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
) -> list[str]:
    values: list[str] = []
    for _ in range(count):
        response: LLMResponse = client.complete(
            prompt, system=system, temperature=temperature, max_tokens=max_tokens
        )
        parsed = parser(response.text)
        values.append(canonical_value(resolve_field({"parsed": parsed}, field)))
    return values


def _validate_count(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be greater than zero")
