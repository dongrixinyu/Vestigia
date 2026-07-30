"""Match multiple externally collected features against saved fingerprints.

Every observed feature must declare its ``prompt_id`` and ``params_hash``.
A non-empty hash must equal the ``parameters_hash`` in a saved fingerprint JSON;
an empty hash searches all parameter configurations for that prompt. All listed
features are jointly considered: a candidate model must have a reference for
every feature and its final distance is their equal-weight mean.
Run from the repository root after filling the values below::

    python examples/predict_distribution.py
"""
from __future__ import annotations

import json
from pathlib import Path

from vestigia import predict_distribution

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_DIRECTORY = REPOSITORY_ROOT / "fingerprints"

# Each item is one observed feature distribution. All items are considered
# together: only models with saved fingerprints for *both* probes are eligible,
# and their per-probe distances are averaged into the final model distance.
# Set ``params_hash`` to "" to allow any saved parameter configuration for the
# corresponding prompt.
OBSERVED_DISTRIBUTIONS = [
    {
        "prompt_id": "favorite_number",
        "params_hash": "",
        "values": [
    "121",
    "163",
    "137",
    "137",
    "137",
    "157",
    "137",
    "137",]
    },
    {
        "prompt_id": "project_success_score",
        "params_hash": "",
        "values": [    "0.3",
    "0.35",
    "0.35",
    "0.3",
    "0.35",
    "0.45",
    "0.35",
    "0.35",
    "0.35"],
    },
    {
        "prompt_id": "outdoor_trip_choice",
        "params_hash": "",
        "values": [
    "forest",
    "lakeside",
    "beach",
    "parse_error",
    "beach"
    ]
    }
    # Add independent probe features to improve model-level identification:
    # {
    #     "prompt_id": "model_identity",
    #     "params_hash": "replace-with-saved-parameters_hash",
    #     "values": ["gpt", "gpt", "null", "gpt"],
    # },
]


def main() -> None:
    result = predict_distribution(
        OBSERVED_DISTRIBUTIONS,
        FINGERPRINT_DIRECTORY,
        # Select a registered distance type for ranking and relative softmax score.
        # Each match exposes its selected metric through ``distance_type`` and
        # its value through ``distance``.
        distance_type="jensen_shannon",
        # distance_type="total_variation",
        softmax_temperature=0.1,
    )

    print("Observed features:", [item["prompt_id"] for item in OBSERVED_DISTRIBUTIONS])
    print(f"\nMatches (sorted by {result.distance_type} distance):")
    headers = ("Rank", "Model", "Distance type", "Distance", "Relative score")  # , "Reference")
    rows = [
        (
            str(rank),
            match.model,
            match.distance_type,
            f"{match.distance:.4f}",
            f"{match.probability:.2%}",
            # match.fingerprint_path.name,
        )
        for rank, match in enumerate(result.matches, start=1)
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row: tuple[str, ...], column_widths: list[int] = widths) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(row, column_widths, strict=True)
        ) + " |"

    print(separator)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)

    print("\nPer-feature distances used in each aggregate result:")
    feature_headers = ("Model", "Prompt", "Distance", "Parameters hash",)  # "Fingerprint")
    feature_rows = [
        (
            match.model,
            feature.prompt_id,
            f"{feature.distance:.4f}",
            feature.params_hash,
            # str(feature.fingerprint_path.relative_to(FINGERPRINT_DIRECTORY)),
        )
        for match in result.matches
        for feature in match.feature_matches
    ]
    feature_widths = [
        max(len(header), *(len(row[index]) for row in feature_rows))
        for index, header in enumerate(feature_headers)
    ]
    feature_separator = "+" + "+".join("-" * (width + 2) for width in feature_widths) + "+"
    print(feature_separator)
    print(format_row(feature_headers, feature_widths))
    print(feature_separator)
    for row in feature_rows:
        print(format_row(row, feature_widths))
    print(feature_separator)
    # print("\nFull JSON result:")
    # print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
