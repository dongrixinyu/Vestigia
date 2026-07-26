from __future__ import annotations

import json

from vestigia.fingerprint import build_fingerprint


def test_build_fingerprint_counts_values_per_identical_request_signature() -> None:
    signature = {
        "digest": "same-request",
        "model": "example-model",
        "provider": "openai_compatible",
        "prompt_id": "favorite_number",
        "prompt": "Pick a number.",
        "temperature": 0.1,
        "max_tokens": 32,
        "system": None,
        "extra_body": {"top_p": 0.9},
    }
    records = [
        {
            "status": "ok",
            "request_signature": signature,
            "parsed": {"first_number": {"value": "76"}},
        }
        for _ in range(10)
    ]
    records.extend(
        {
            "status": "ok",
            "request_signature": signature,
            "parsed": {"first_number": {"value": "34"}},
        }
        for _ in range(1)
    )
    records.append({"status": "error"})

    fingerprint = build_fingerprint(records, field="parsed.first_number.value")

    assert fingerprint["format"] == "vestigia.empirical-fingerprint.v1"
    summary = fingerprint["fingerprints"][0]
    assert summary["sample_count"] == 11
    assert summary["values"] == [
        {"value": "76", "count": 10, "proportion": 10 / 11},
        {"value": "34", "count": 1, "proportion": 1 / 11},
    ]
    assert json.loads(json.dumps(fingerprint, ensure_ascii=False)) == fingerprint
