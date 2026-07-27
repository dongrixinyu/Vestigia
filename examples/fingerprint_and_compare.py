"""Build, save, load, and compare an LLM fingerprint using only Vestigia APIs.

This script selects Vestigia's built-in ``favorite_number`` probe, sends its
first fixed wording repeatedly to a reference model, saves its response
fingerprint, then samples a candidate model under precisely the saved request
conditions and prints the comparison report.

Run from the repository root::

    export REFERENCE_BASE_URL="https://gateway.example.com/v1"
    export REFERENCE_API_KEY="reference-key"
    export REFERENCE_MODEL="reference-model"
    export CANDIDATE_BASE_URL="https://other-gateway.example.com/v1"
    export CANDIDATE_API_KEY="candidate-key"
    export CANDIDATE_MODEL="candidate-model"
    python examples/fingerprint_and_compare.py

Optional variables:

* ``REFERENCE_PROVIDER`` / ``CANDIDATE_PROVIDER``: ``openai_compatible``
  (default) or ``anthropic``;
* ``FINGERPRINT_PATH``: output path (default
  ``fingerprints/favorite-number-reference.json``);
* ``REFERENCE_COUNT``: reference calls (default 50);
* ``CANDIDATE_COUNT``: candidate calls (default 20).

No manual LLM client, request loop, distribution aggregation, or JSON
serialisation is implemented here: all of that is handled by Vestigia.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from vestigia import create_fingerprint, load_fingerprint, verify_fingerprint


def required(name: str) -> str:
    """Read a mandatory environment variable with an actionable error."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set the required environment variable {name}.")
    return value


def main() -> None:
    fingerprint_path = Path(
        os.environ.get(
            "FINGERPRINT_PATH", "fingerprints/favorite-number-reference.json"
        )
    )

    # 1. Select one built-in prompt and call the reference model repeatedly.
    # Vestigia uses favorite_number's parser; the field below records the
    # normalized first number instead of the entire response text.
    reference = create_fingerprint(
        base_url=required("REFERENCE_BASE_URL"),
        api_key=required("REFERENCE_API_KEY"),
        model=required("REFERENCE_MODEL"),
        provider=os.environ.get("REFERENCE_PROVIDER", "openai_compatible"),
        prompt_id="favorite_number",
        variant_index=0,
        field="parsed.first_number.value",
        count=int(os.environ.get("REFERENCE_COUNT", "50")),
        temperature=0.1,
        max_tokens=64,
        output=fingerprint_path,
        subset_size=20,
        resamples=1_000,
        seed=42,
    )
    print(f"Reference fingerprint saved to: {fingerprint_path}")
    print("Reference distribution:")
    print(json.dumps(reference.distribution, ensure_ascii=False, indent=2))

    # 2. Load the persisted reference (as a separate process would), then
    # repeat its exact prompt and sampling controls against the candidate.
    # The built-in probe parser is recovered automatically by verify_fingerprint.
    persisted_reference = load_fingerprint(fingerprint_path)
    result = verify_fingerprint(
        persisted_reference,
        base_url=required("CANDIDATE_BASE_URL"),
        api_key=required("CANDIDATE_API_KEY"),
        model=required("CANDIDATE_MODEL"),
        provider=os.environ.get("CANDIDATE_PROVIDER", "openai_compatible"),
        count=int(os.environ.get("CANDIDATE_COUNT", "20")),
    )

    # 3. The full report includes candidate distribution, TV distances,
    # reference acceptance thresholds, and the final match decision.
    print("\nCandidate comparison report:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nMatches reference: {result.matches_reference}")


if __name__ == "__main__":
    main()
