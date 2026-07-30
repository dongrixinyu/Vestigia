"""
Collect and save favorite-number / project-success-score fingerprints for
multiple model endpoints.

Each endpoint/model/prompt combination is sampled independently and saved under
the repository-level ``fingerprints`` directory.

Run after filling the connection settings below::

    python examples/fingerprint_and_compare.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vestigia import create_fingerprint, save_fingerprint, get_request_params

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_DIRECTORY = REPOSITORY_ROOT / "fingerprints"

# Number of actual model calls for every endpoint/model/prompt combination.
SAMPLE_COUNT = 5

# Every prompt_id to collect.
PROMPT_IDS = [
    # "favorite_number",
    # "project_success_score",
    # "outdoor_trip_choice",
    "model_identity",
]


def safe_path_component(value: str) -> str:
    """Convert endpoint/model identifiers to safe directory names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


# Add one item for each direct provider or relay.
#
# Each endpoint may contain multiple models. ``provider`` controls only the
# endpoint wire protocol; a relay serving an Anthropic, DeepSeek, Kimi, or
# other model through OpenAI-compatible APIs must still use
# provider="openai_compatible".
MODEL_ENDPOINTS = []

def main() -> None:
    for endpoint in MODEL_ENDPOINTS:
        endpoint_name = str(endpoint["name"])
        provider = str(endpoint.get("provider", "openai_compatible"))

        for model_config in endpoint["models"]:
            model_name = str(model_config["model"])

            # Put each endpoint/model combination in a separate directory.
            # Prompt-specific filenames are then produced by save_fingerprint.
            output_directory = (
                FINGERPRINT_DIRECTORY
                / safe_path_component(endpoint_name)
                / safe_path_component(model_name)
            )
            output_directory.mkdir(parents=True, exist_ok=True)

            for prompt_id in PROMPT_IDS:
                print(
                    "\nCollecting distribution:"
                    f" endpoint={endpoint_name}"
                    f" model={model_name}"
                    f" prompt_id={prompt_id}"
                )

                fingerprint = create_fingerprint(
                    base_url=str(endpoint["base_url"]),
                    api_key=str(endpoint["api_key"]),
                    model=model_name,
                    provider=provider,
                    endpoint=endpoint.get("endpoint"),
                    prompt_id=prompt_id,
                    variant_index=0,
                    count=SAMPLE_COUNT,
                    request_params=model_config.get("request_params"),
                    output=output_directory,
                )

                fingerprint_path = save_fingerprint(
                    fingerprint,
                    output_directory,
                    prompt_id=prompt_id,
                )

                print(f"Saved: {fingerprint_path}")
                print(
                    json.dumps(
                        fingerprint.distribution,
                        ensure_ascii=False,
                        indent=2,
                    )
                )


if __name__ == "__main__":
    main()
