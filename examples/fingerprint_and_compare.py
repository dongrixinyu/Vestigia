"""Collect and save favorite-number fingerprints for multiple model endpoints.

Add one dictionary to ``MODEL_ENDPOINTS`` for every direct provider or relay.
Each endpoint is sampled independently and saved under the repository-level
``fingerprints`` directory. This example only collects distributions; it does
not identify or compare models.

Run after filling the connection settings below::

    python examples/fingerprint_and_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vestigia import create_fingerprint, save_fingerprint

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_DIRECTORY = REPOSITORY_ROOT / "fingerprints"

# Number of actual model calls per endpoint. This is the only sampling-count
# setting exposed by create_fingerprint.
SAMPLE_COUNT = 50

# Add one item for each model and/or relay that should have a distribution.
# ``provider`` controls only the endpoint's wire protocol. A relay serving an
# Anthropic, DeepSeek, Kimi, or other model through OpenAI-compatible APIs must
# therefore still use ``provider="openai_compatible"``.
MODEL_ENDPOINTS: list[dict[str, Any]] = [
    {
        "name": "deepseek-direct",
        "base_url": "https://api.deepseek.com",
        "api_key": "your-deepseek-api-key",
        "model": "deepseek-v4-pro",
        "provider": "openai_compatible",
        "request_params": {
            "temperature": 0.1,
            "max_tokens": 1024,
        },
    },
    # Example: OpenAI-compatible relay that exposes a Claude model.
    # {
    #     "name": "claude-through-relay-a",
    #     "base_url": "https://relay-a.example.com/v1",
    #     "api_key": "your-relay-a-api-key",
    #     "model": "claude-sonnet-4-6",
    #     "provider": "openai_compatible",
    #     "request_params": {
    #         "temperature": 0.1,
    #         "max_tokens": 1024,
    #         "extra_body": {"top_p": 0.9},
    #     },
    # },
    # Example: direct Anthropic Messages API.
    # {
    #     "name": "claude-direct",
    #     "base_url": "https://api.anthropic.com",
    #     "api_key": "your-anthropic-api-key",
    #     "model": "claude-sonnet-4-6",
    #     "provider": "anthropic",
    #     "request_params": {"temperature": 0.1, "max_tokens": 1024},
    # },
]


def main() -> None:
    for endpoint in MODEL_ENDPOINTS:
        name = str(endpoint["name"])
        print(f"\nCollecting favorite_number distribution: {name}")
        fingerprint = create_fingerprint(
            base_url=str(endpoint["base_url"]),
            api_key=str(endpoint["api_key"]),
            model=str(endpoint["model"]),
            provider=str(endpoint.get("provider", "openai_compatible")),
            endpoint=endpoint.get("endpoint"),
            prompt_id="favorite_number",
            variant_index=0,
            field="parsed.first_number.value",
            count=SAMPLE_COUNT,
            request_params=endpoint.get("request_params"),
            output=FINGERPRINT_DIRECTORY,
        )
        fingerprint_path = save_fingerprint(
            fingerprint, FINGERPRINT_DIRECTORY, prompt_id="favorite_number"
        )
        print(f"Saved: {fingerprint_path}")
        print(json.dumps(fingerprint.distribution, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
