"""Statistical validation and comparison of empirical LLM fingerprints."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from statistics import fmean
from typing import Any

from vestigia.fingerprint import canonical_value, resolve_field


def distribution(values: Iterable[str]) -> dict[str, float]:
    """Return a normalized categorical distribution."""
    counts = Counter(values)
    total = sum(counts.values())
    if not total:
        raise ValueError("a distribution requires at least one successful sample")
    return {value: count / total for value, count in counts.items()}


def total_variation_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return total-variation distance (0 identical, 1 disjoint)."""
    ret = (
        sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in left.keys() | right.keys()) / 2
    )
    return ret

def jensen_shannon_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return Jensen-Shannon distance in [0, 1], using base-2 logarithms."""
    midpoint = {
        key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2 for key in left.keys() | right.keys()
    }

    def kl(source: Mapping[str, float]) -> float:
        return sum(
            probability * math.log2(probability / midpoint[key])
            for key, probability in source.items()
        )

    ret = math.sqrt((kl(left) + kl(right)) / 2)
    return ret


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated percentile, where ``fraction`` is in [0, 1]."""
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def validate_stability(
    values: Sequence[str],
    *,
    sample_size: int = 20,
    resamples: int = 1_000,
    seed: int | None = 0,
    max_p95_tv_distance: float = 0.20,
) -> dict[str, Any]:
    """Estimate whether random subsets preserve a full-run distribution.

    Each resample is drawn *without replacement* from ``values`` and compared
    to the full-run empirical distribution. A fingerprint is accepted when its
    95th-percentile total-variation distance is within the supplied threshold.
    This is a Monte-Carlo estimate, not a claim about every possible subset.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be greater than zero")
    if sample_size > len(values):
        raise ValueError("sample_size cannot exceed the successful sample count")
    if resamples < 1:
        raise ValueError("resamples must be greater than zero")
    if not 0 <= max_p95_tv_distance <= 1:
        raise ValueError("max_p95_tv_distance must be between zero and one")

    reference = distribution(values)
    generator = random.Random(seed)
    distances = [
        (
            total_variation_distance(
                distribution(subset := generator.sample(values, sample_size)), reference
            ),
            jensen_shannon_distance(distribution(subset), reference),
        )
        for _ in range(resamples)
    ]
    tv_distances = [distance[0] for distance in distances]
    js_distances = [distance[1] for distance in distances]
    tv_p95 = percentile(tv_distances, 0.95)
    return {
        "successful_sample_count": len(values),
        "subset_size": sample_size,
        "resamples": resamples,
        "seed": seed,
        "reference_distribution": reference,
        "total_variation_distance": _summary(tv_distances),
        "jensen_shannon_distance": _summary(js_distances),
        "max_p95_tv_distance": max_p95_tv_distance,
        "reliable": tv_p95 <= max_p95_tv_distance,
    }


def successful_values(records: Iterable[Mapping[str, Any]], field: str) -> list[str]:
    """Extract a canonical feature value from every successful collection row."""
    return [
        canonical_value(resolve_field(record, field))
        for record in records
        if record.get("status") == "ok"
    ]


def compare_distributions(left: Sequence[str], right: Sequence[str]) -> dict[str, float]:
    """Calculate complementary distances between two empirical fingerprints."""
    left_distribution = distribution(left)
    right_distribution = distribution(right)
    return {
        "total_variation_distance": total_variation_distance(left_distribution, right_distribution),
        "jensen_shannon_distance": jensen_shannon_distance(left_distribution, right_distribution),
    }


def log_length_bucket(length: int) -> dict[str, int]:
    """Map a character length to a power-of-two histogram bucket.

    Positive lengths are grouped as ``[1, 2)``, ``[2, 4)``, ``[4, 8)`` and
    so on. Zero has its own bucket. This prevents a few very long answers from
    making an exact-length histogram sparse and unstable.
    """
    if length < 0:
        raise ValueError("text length must not be negative")
    if length == 0:
        return {"lower": 0, "upper_exclusive": 1}
    lower = 1 << (length.bit_length() - 1)
    return {"lower": lower, "upper_exclusive": lower * 2}


def log_length_values(values: Iterable[int]) -> list[str]:
    """Return canonical power-of-two bucket values for text lengths."""
    return [canonical_value(log_length_bucket(value)) for value in values]


def text_length_summary(values: Sequence[int]) -> dict[str, float]:
    """Return descriptive statistics for raw response lengths in Unicode characters."""
    if not values:
        raise ValueError("text length statistics require at least one sample")
    mean = fmean(values)
    variance = fmean((value - mean) ** 2 for value in values)
    return {
        "mean": mean,
        "standard_deviation": math.sqrt(variance),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def validate_length_distribution(
    values: Sequence[int],
    *,
    sample_size: int = 20,
    resamples: int = 1_000,
    seed: int | None = 0,
    max_p95_tv_distance: float = 0.20,
) -> dict[str, Any]:
    """Validate a power-of-two text-length histogram using subset TV distance."""
    buckets = log_length_values(values)
    result = validate_stability(
        buckets,
        sample_size=sample_size,
        resamples=resamples,
        seed=seed,
        max_p95_tv_distance=max_p95_tv_distance,
    )
    return {
        "bucket_scheme": "power_of_two_characters",
        "statistics": text_length_summary(values),
        "distribution": result["reference_distribution"],
        "stability": {
            key: value
            for key, value in result.items()
            if key not in {"successful_sample_count", "reference_distribution"}
        },
    }


def validate_mean_stability(
    values: Sequence[int],
    *,
    sample_size: int = 20,
    resamples: int = 1_000,
    seed: int | None = 0,
    max_p95_relative_mean_delta: float = 0.20,
) -> dict[str, Any]:
    """Validate whether subset mean text length is stable relative to the full run."""
    if sample_size < 1:
        raise ValueError("sample_size must be greater than zero")
    if sample_size > len(values):
        raise ValueError("sample_size cannot exceed the successful sample count")
    if resamples < 1:
        raise ValueError("resamples must be greater than zero")
    if not 0 <= max_p95_relative_mean_delta:
        raise ValueError("max_p95_relative_mean_delta must not be negative")

    reference_mean = fmean(values)
    generator = random.Random(seed)
    deltas = [
        abs(fmean(generator.sample(values, sample_size)) - reference_mean) for _ in range(resamples)
    ]
    p95 = percentile(deltas, 0.95)
    relative_p95 = p95 / reference_mean if reference_mean else 0.0 if p95 == 0 else math.inf
    return {
        "statistics": text_length_summary(values),
        "subset_size": sample_size,
        "resamples": resamples,
        "seed": seed,
        "absolute_mean_delta": _summary(deltas),
        "p95_relative_mean_delta": relative_p95,
        "max_p95_relative_mean_delta": max_p95_relative_mean_delta,
        "reliable": relative_p95 <= max_p95_relative_mean_delta,
    }


def _summary(distances: Sequence[float]) -> dict[str, float]:
    return {
        "mean": fmean(distances),
        "p50": percentile(distances, 0.50),
        "p95": percentile(distances, 0.95),
        "max": max(distances),
    }
