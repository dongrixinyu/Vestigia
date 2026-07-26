from __future__ import annotations

import pytest

from vestigia.validation import (
    compare_distributions,
    log_length_bucket,
    successful_values,
    validate_length_distribution,
    validate_stability,
)


def test_stability_is_reproducible_and_accepts_a_stable_distribution() -> None:
    values = ["76"] * 25 + ["34"] * 25

    result = validate_stability(
        values, sample_size=20, resamples=200, seed=123, max_p95_tv_distance=0.25
    )

    assert result["reference_distribution"] == {"76": 0.5, "34": 0.5}
    assert result["total_variation_distance"]["p95"] <= 0.25
    assert result["reliable"] is True
    assert result == validate_stability(
        values, sample_size=20, resamples=200, seed=123, max_p95_tv_distance=0.25
    )


def test_comparison_separates_distinct_model_distributions() -> None:
    claude = ["76"] * 25 + ["34"] * 25
    kimi = ["76"] * 5 + ["36"] * 45

    distances = compare_distributions(claude, kimi)

    assert distances["total_variation_distance"] == pytest.approx(0.9)
    assert distances["jensen_shannon_distance"] > 0.6


def test_length_distribution_uses_power_of_two_buckets() -> None:
    values = [1, 2, 3, 4, 7, 8, 15, 16]

    result = validate_length_distribution(values, sample_size=4, resamples=20, seed=1)

    assert log_length_bucket(7) == {"lower": 4, "upper_exclusive": 8}
    assert result["bucket_scheme"] == "power_of_two_characters"
    assert result["distribution"] == {
        '{"lower":1,"upper_exclusive":2}': 0.125,
        '{"lower":2,"upper_exclusive":4}': 0.25,
        '{"lower":4,"upper_exclusive":8}': 0.25,
        '{"lower":8,"upper_exclusive":16}': 0.25,
        '{"lower":16,"upper_exclusive":32}': 0.125,
    }


def test_successful_values_ignores_errors_and_canonicalizes_feature() -> None:
    records = [
        {"status": "ok", "parsed": {"first_number": {"value": "76"}}},
        {"status": "error", "parsed": {"first_number": {"value": "34"}}},
    ]

    assert successful_values(records, "parsed.first_number.value") == ['"76"']
