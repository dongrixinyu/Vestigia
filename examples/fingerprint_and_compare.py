"""Build, save, load, and compare an LLM fingerprint using only Vestigia APIs.

This script selects Vestigia's built-in ``favorite_number`` probe, sends its
first fixed wording repeatedly to a reference model, saves its response
fingerprint, then samples a candidate model under precisely the saved request
conditions and prints the comparison report.

Run after filling the connection constants below::

    python examples/fingerprint_and_compare.py

No manual LLM client, request loop, distribution aggregation, or JSON
serialisation is implemented here: all of that is handled by Vestigia.
"""
from __future__ import annotations

import json
from pathlib import Path

from vestigia import create_fingerprint, load_fingerprint, save_fingerprint, verify_fingerprint

# Fill in the reference model connection.
LLM_BASE_URL = "https://gateway.example.com/v1"
LLM_API_KEY = "your-reference-api-key"
LLM_MODEL = "your-reference-model"
LLM_PROVIDER = "openai_compatible"  # Endpoint wire protocol; or "anthropic".



# Fill in the model to compare against the reference.
CANDIDATE_LLM_BASE_URL = "https://gateway.example.com/v1"
CANDIDATE_LLM_API_KEY = "your-candidate-api-key"
CANDIDATE_LLM_MODEL = "your-candidate-model"
CANDIDATE_LLM_PROVIDER = "openai_compatible"  # Or "anthropic".

MAX_TOKENS = 1024
REFERENCE_COUNT = 5
CANDIDATE_COUNT = 20
FINGERPRINT_DIRECTORY = Path("fingerprints")


def main() -> None:
    # 1. Select one built-in prompt and call the reference model repeatedly.
    # Vestigia uses favorite_number's parser; the field below records the
    # normalized first number instead of the entire response text.
    reference = create_fingerprint(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        provider=LLM_PROVIDER,
        prompt_id="favorite_number",
        variant_index=0,
        field="parsed.first_number.value",
        count=REFERENCE_COUNT,
        temperature=0.1,
        max_tokens=MAX_TOKENS,
        # ``output`` is a directory. Vestigia chooses the configuration-specific
        # filename: {model}__{prompt_id}__{params_hash}.json.
        output=FINGERPRINT_DIRECTORY,
    )
    # save_fingerprint returns the exact canonical path chosen internally.
    # Calling it again writes the same configuration-specific file and lets this
    # example retain the path for a later load.
    fingerprint_path = save_fingerprint(
        reference, FINGERPRINT_DIRECTORY, prompt_id="favorite_number"
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
        base_url=CANDIDATE_LLM_BASE_URL,
        api_key=CANDIDATE_LLM_API_KEY,
        model=CANDIDATE_LLM_MODEL,
        provider=CANDIDATE_LLM_PROVIDER,
        count=CANDIDATE_COUNT,
    )

    # 3. The full report includes candidate distribution, TV distances,
    # reference acceptance thresholds, and the final match decision.
    print("\nCandidate comparison report:")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"\nMatches reference: {result.matches_reference}")


if __name__ == "__main__":
    main()
