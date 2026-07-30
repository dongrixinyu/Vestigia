"""Collect behavioral fingerprints from an unknown model exposed by a relay.

Set the relay connection values below, then run from the repository root:

    PYTHONPATH=src python examples/collect_unknown_model_fingerprint.py

The relay's advertised model name is stored as metadata only. The script does
not assume that name identifies the underlying model. Each selected probe is
sampled independently and saved under ``unknown_fingerprints/<relay>/<model>/``.
They are then jointly compared with the known reference library under
``fingerprints/`` using ``predict_distribution()``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from vestigia import create_fingerprint, get_request_params, predict_distribution

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_DIRECTORY = REPOSITORY_ROOT / "fingerprints"
# Existing, known-model reference fingerprints used by predict_distribution().
REFERENCE_FINGERPRINT_DIRECTORY = FINGERPRINT_DIRECTORY
# Keep newly collected unknown-model fingerprints out of the reference library;
# otherwise prediction would select the candidate itself with distance zero.
CANDIDATE_FINGERPRINT_DIRECTORY = REPOSITORY_ROOT / "unknown_fingerprints"

# Connection settings for the relay. Never commit an actual API key.
RELAY_NAME = "unknown-relay"
BASE_URL = "https://relay.example.com/v1"
API_KEY = "replace-with-relay-api-key"
PROVIDER = "openai_compatible"
ENDPOINT: str | None = None  # Set only when the relay uses a non-standard route.

# This is the model name accepted by the relay. It is not treated as proof of
# the underlying model identity; it only names the collected fingerprint files.
RELAY_MODEL_NAME = "relay-advertised-model-name"

# Select one fixed, versioned request-parameter profile. Use ``kimi_v1`` only
# for a Kimi relay; it fixes top_p=0.95. See doc/request_params.md.
REQUEST_PARAM_PRESET = "standard_v1"

# Number of calls made for every probe. Increasing this usually improves the
# empirical distribution and stability estimate.
SAMPLE_COUNT = 20

# Multiple probe distributions are needed for model-level identification.
PROMPT_IDS = (
    "favorite_number",
    "project_success_score",
    "outdoor_trip_choice",
)


def safe_path_component(value: str) -> str:
    """Convert relay/model names into one safe directory component."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._") or "unknown"


def main() -> None:
    if API_KEY == "replace-with-relay-api-key":
        raise ValueError("set API_KEY before collecting fingerprints")
    if RELAY_MODEL_NAME == "relay-advertised-model-name":
        raise ValueError("set RELAY_MODEL_NAME before collecting fingerprints")

    output_directory = (
        CANDIDATE_FINGERPRINT_DIRECTORY
        / safe_path_component(RELAY_NAME)
        / safe_path_component(RELAY_MODEL_NAME)
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    observed_distributions: list[dict[str, object]] = []
    for prompt_id in PROMPT_IDS:
        print(
            "\nCollecting distribution:"
            f" relay={RELAY_NAME}"
            f" model={RELAY_MODEL_NAME}"
            f" prompt_id={prompt_id}"
            f" preset={REQUEST_PARAM_PRESET}"
        )
        fingerprint = create_fingerprint(
            base_url=BASE_URL,
            api_key=API_KEY,
            model=RELAY_MODEL_NAME,
            provider=PROVIDER,
            endpoint=ENDPOINT,
            prompt_id=prompt_id,
            variant_index=0,
            count=SAMPLE_COUNT,
            request_params=get_request_params(REQUEST_PARAM_PRESET),
            output=output_directory,
        )
        observed_distributions.append(
            {
                "prompt_id": prompt_id,
                # An empty hash is a wildcard: compare with every saved
                # parameter configuration for this probe in the reference
                # library. The selected reference hash is reported in each
                # feature match.
                "params_hash": "",
                "values": list(fingerprint.values),
            }
        )
        print("Distribution:")
        print(json.dumps(fingerprint.distribution, ensure_ascii=False, indent=2))

    print(f"\nCandidate fingerprints saved under: {output_directory}")
    print(f"Comparing against reference library: {REFERENCE_FINGERPRINT_DIRECTORY}")
    result = predict_distribution(
        observed_distributions,
        REFERENCE_FINGERPRINT_DIRECTORY,
        distance_type="total_variation",
        softmax_temperature=0.1,
    )

    print(f"\nPredicted models (sorted by {result.distance_type}):")
    for rank, match in enumerate(result.matches, start=1):
        print(
            f"{rank}. model={match.model}"
            f" distance={match.distance:.4f}"
            f" probability={match.probability:.2%}"
        )
        for feature in match.feature_matches:
            print(
                f"   - prompt_id={feature.prompt_id}"
                f" distance={feature.distance:.4f}"
                f" params_hash={feature.params_hash}"
                f" reference={feature.fingerprint_path}"
            )

    # For APIs, logs, or downstream programs:
    # print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
