"""One-call workflows for creating and verifying persisted model fingerprints."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vestigia.identify import (
    FingerprintTestResult,
    ModelFingerprint,
    Parser,
    build_model_fingerprint,
    test_model_against_fingerprint,
)
from vestigia.llm import LLMClient, LLMConfig


def text_parser(text: str) -> dict[str, str]:
    """Default feature parser: compare the complete response text."""
    return {"text": text}


def create_fingerprint(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    output: str | Path | None = None,
    provider: str = "openai_compatible",
    endpoint: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    reasoning: Mapping[str, Any] | None = None,
    reasoning_effort: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    parser: Parser = text_parser,
    field: str = "parsed.text",
    count: int = 50,
    subset_size: int = 20,
    resamples: int = 1_000,
    seed: int | None = 0,
) -> ModelFingerprint:
    """Call one endpoint repeatedly and optionally save its stable distribution.

    ``base_url``, ``api_key`` and ``model`` are the only required connection
    values. Sampling controls accepted by the target API belong in
    ``temperature``, ``max_tokens`` and ``extra_body``. The saved JSON can be
    passed directly to :func:`verify_fingerprint` after loading with
    :func:`load_fingerprint`.
    """
    config = LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        reasoning=reasoning,
        reasoning_effort=reasoning_effort,
        extra_body=extra_body or {},
        extra_headers=extra_headers or {},
    )
    with LLMClient(config) as client:
        fingerprint = build_model_fingerprint(
            client,
            prompt,
            parser,
            count=count,
            field=field,
            system=system,
            subset_size=subset_size,
            resamples=resamples,
            seed=seed,
        )
    if output is not None:
        save_fingerprint(fingerprint, output)
    return fingerprint


def verify_fingerprint(
    fingerprint: ModelFingerprint | Mapping[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "openai_compatible",
    endpoint: str | None = None,
    extra_body: Mapping[str, Any] | None = None,
    extra_headers: Mapping[str, str] | None = None,
    parser: Parser = text_parser,
    count: int = 20,
) -> FingerprintTestResult:
    """Repeat the reference request against another model and compare it.

    The reference prompt, system instruction, token limit, temperature and
    feature field are reused automatically. ``extra_body`` must contain the
    same sampling controls as the reference by default. Pass ``extra_body``
    only to explicitly override them; a mismatch raises ``ValueError``.
    """
    reference = _coerce_fingerprint(fingerprint)
    config = LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        temperature=reference.temperature,
        max_tokens=reference.max_tokens,
        top_p=reference.request_configuration.get("top_p"),
        top_k=reference.request_configuration.get("top_k"),
        presence_penalty=reference.request_configuration.get("presence_penalty"),
        frequency_penalty=reference.request_configuration.get("frequency_penalty"),
        reasoning=reference.request_configuration.get("reasoning"),
        reasoning_effort=reference.request_configuration.get("reasoning_effort"),
        extra_body=(
            extra_body
            if extra_body is not None
            else dict(reference.request_configuration.get("extra_body", {}))
        ),
        extra_headers=extra_headers or {},
    )
    with LLMClient(config) as client:
        return test_model_against_fingerprint(client, reference, parser, count=count)


def save_fingerprint(fingerprint: ModelFingerprint, output: str | Path) -> None:
    """Persist a reference fingerprint as readable UTF-8 JSON."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint.to_dict(), ensure_ascii=False, indent=2) + "\n", "utf-8")


def load_fingerprint(input_path: str | Path) -> ModelFingerprint:
    """Load a fingerprint previously written by :func:`save_fingerprint`."""
    data = json.loads(Path(input_path).read_text("utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("fingerprint JSON must be an object")
    return _coerce_fingerprint(data)


def _coerce_fingerprint(value: ModelFingerprint | Mapping[str, Any]) -> ModelFingerprint:
    if isinstance(value, ModelFingerprint):
        return value
    required = {
        "model", "provider", "prompt", "system", "temperature", "max_tokens",
        "request_configuration", "field", "values", "distribution", "text_length", "stability",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"fingerprint is missing fields: {', '.join(sorted(missing))}")
    return ModelFingerprint(
        model=str(value["model"]),
        provider=str(value["provider"]),
        prompt=str(value["prompt"]),
        system=value["system"] if isinstance(value["system"], str) else None,
        temperature=value["temperature"] if isinstance(value["temperature"], (int, float)) else None,
        max_tokens=value["max_tokens"] if isinstance(value["max_tokens"], int) else None,
        request_configuration=dict(value["request_configuration"]),
        field=str(value["field"]),
        values=tuple(str(item) for item in value["values"]),
        distribution=dict(value["distribution"]),
        text_length=dict(value["text_length"]),
        stability=dict(value["stability"]),
    )
