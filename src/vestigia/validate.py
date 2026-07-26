"""CLI for validating fingerprint stability and comparing model distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vestigia.validation import compare_distributions, successful_values, validate_stability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate subset stability and compare empirical LLM fingerprints."
    )
    parser.add_argument("--input", type=Path, required=True, help="Primary collector JSONL file")
    parser.add_argument("--compare-input", type=Path, help="Optional second collector JSONL file")
    parser.add_argument(
        "--field",
        default="parsed.first_number.value",
        help="Dotted record field used as the categorical feature",
    )
    parser.add_argument("--sample-size", type=int, default=20, help="Subset size (default: 20)")
    parser.add_argument(
        "--resamples", type=int, default=1_000, help="Random subsets (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="PRNG seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--max-p95-tv-distance",
        type=float,
        default=0.20,
        help="Reliability threshold for subset-vs-full TV distance (default: 0.20)",
    )
    parser.add_argument("--output", type=Path, help="Write JSON report here; defaults to stdout")
    return parser


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number}: a JSONL record must be an object")
        records.append(record)
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    primary = successful_values(load_records(args.input), args.field)
    report: dict[str, Any] = {
        "format": "vestigia.fingerprint-validation.v1",
        "field": args.field,
        "primary": {
            "input": str(args.input),
            "stability": validate_stability(
                primary,
                sample_size=args.sample_size,
                resamples=args.resamples,
                seed=args.seed,
                max_p95_tv_distance=args.max_p95_tv_distance,
            ),
        },
    }
    if args.compare_input:
        secondary = successful_values(load_records(args.compare_input), args.field)
        secondary_stability = validate_stability(
            secondary,
            sample_size=args.sample_size,
            resamples=args.resamples,
            seed=args.seed,
            max_p95_tv_distance=args.max_p95_tv_distance,
        )
        distances = compare_distributions(primary, secondary)
        report["comparison"] = {
            "input": str(args.compare_input),
            "secondary_stability": secondary_stability,
            "between_model": distances,
            "distinguishable": (
                report["primary"]["stability"]["reliable"]
                and secondary_stability["reliable"]
                and distances["total_variation_distance"]
                > max(
                    report["primary"]["stability"]["total_variation_distance"]["p95"],
                    secondary_stability["total_variation_distance"]["p95"],
                )
            ),
        }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except ValueError as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        return
    print(text, end="")


if __name__ == "__main__":
    main()
