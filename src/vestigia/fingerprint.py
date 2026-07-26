"""Build reproducible empirical fingerprints from collected JSONL records."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


def resolve_field(record: Mapping[str, Any], field: str) -> Any:
    """Return a dotted field from a collection record.

    A missing field is represented by ``None`` so missing values are counted
    explicitly instead of silently being discarded from a fingerprint.
    """
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def canonical_value(value: Any) -> str:
    """Encode a feature value as a stable JSON distribution key."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_fingerprint(
    records: Iterable[Mapping[str, Any]], *, field: str = "parsed"
) -> dict[str, Any]:
    """Summarize successful samples as an empirical value distribution.

    Records are grouped by the request-signature digest.  Consequently only
    calls with the identical model, prompt, generation parameters, system
    instruction and extra body fields are compared in one distribution.
    """
    groups: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        signature = record.get("request_signature")
        if not isinstance(signature, Mapping) or not isinstance(signature.get("digest"), str):
            continue

        digest = signature["digest"]
        group = groups.setdefault(
            digest,
            {
                "request_signature": dict(signature),
                "sample_count": 0,
                "distribution": Counter[str](),
            },
        )
        group["sample_count"] += 1
        group["distribution"][canonical_value(resolve_field(record, field))] += 1

    fingerprints: list[dict[str, Any]] = []
    for digest in sorted(groups):
        group = groups[digest]
        sample_count = group["sample_count"]
        distribution: Counter[str] = group["distribution"]
        values = [
            {"value": json.loads(value), "count": count, "proportion": count / sample_count}
            for value, count in distribution.items()
        ]
        values.sort(key=lambda item: (-item["count"], canonical_value(item["value"])))
        fingerprints.append(
            {
                "request_signature": group["request_signature"],
                "sample_count": sample_count,
                "field": field,
                "values": values,
            }
        )

    return {"format": "vestigia.empirical-fingerprint.v1", "fingerprints": fingerprints}
