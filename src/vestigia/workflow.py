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
from vestigia.prompts import DEFAULT_PROMPTS, PromptTemplate


def text_parser(text: str) -> dict[str, str]:
    """Default feature parser: compare the complete response text."""
    return {"text": text}


def create_fingerprint(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_id: str,
    variant_index: int = 0,
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
    parser: Parser | None = None,
    field: str | None = None,
    count: int = 50,
    subset_size: int = 20,
    resamples: int = 1_000,
    seed: int | None = 0,
) -> ModelFingerprint:
    """Sample one built-in probe repeatedly and save its response distribution.

    ``prompt_id`` selects a probe from :mod:`vestigia.prompts`; arbitrary prompt
    text is deliberately not accepted. ``variant_index`` selects one fixed
    wording from that probe, which is then used for every one of the ``count``
    calls. The probe's parser is used automatically unless explicitly
    overridden.

    ``base_url``, ``api_key`` and ``model`` are the required connection values.
    Sampling controls accepted by the target API belong in ``temperature``,
    ``max_tokens`` and ``extra_body``. The saved JSON can be passed directly to
    :func:`verify_fingerprint` after loading with :func:`load_fingerprint`.
    """
    selected_prompt, selected_parser, length_source = _select_prompt(
        prompt_id=prompt_id,
        variant_index=variant_index,
        parser=parser,
    )
    selected_field = field or "parsed"
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
            selected_prompt,
            selected_parser,
            count=count,
            field=selected_field,
            system=system,
            subset_size=subset_size,
            resamples=resamples,
            seed=seed,
            length_source=length_source,
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
    parser: Parser | None = None,
    count: int = 20,
) -> FingerprintTestResult:
    """Repeat the reference request against another model and compare it.

    The reference prompt, system instruction, token limit, temperature and
    feature field are reused automatically. The probe parser is recovered from
    the built-in prompt catalog; pass ``parser`` only to explicitly override
    it. ``extra_body`` must contain the same sampling controls as the reference
    default. Pass ``extra_body`` only to explicitly override them; a mismatch
    raises ``ValueError``.
    """
    reference = _coerce_fingerprint(fingerprint)
    selected_parser = parser or _parser_for_fingerprint(reference)
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
        return test_model_against_fingerprint(client, reference, selected_parser, count=count)


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


def _parser_for_fingerprint(fingerprint: ModelFingerprint) -> Parser:
    """Recover the parser for the built-in probe used by a fingerprint."""
    matching_templates = [
        template
        for template in DEFAULT_PROMPTS
        if fingerprint.prompt in template.variants
    ]
    if len(matching_templates) != 1:
        raise ValueError(
            "fingerprint prompt is not uniquely represented in the built-in prompt catalog; "
            "pass parser explicitly"
        )
    return matching_templates[0].parser


def _select_prompt(
    *,
    prompt_id: str,
    variant_index: int,
    parser: Parser | None,
) -> tuple[str, Parser, str]:
    """Resolve one fixed wording and parser from the built-in probe catalog."""
    if variant_index < 0:
        raise ValueError("variant_index must not be negative")

    template = _prompt_template(prompt_id)
    try:
        selected_prompt = template.variants[variant_index]
    except IndexError as error:
        raise ValueError(
            f"variant_index {variant_index} is out of range for prompt_id {prompt_id!r}; "
            f"choose 0 through {len(template.variants) - 1}"
        ) from error
    return selected_prompt, parser or template.parser, template.length_source


def _prompt_template(prompt_id: str) -> PromptTemplate:
    """Return a built-in probe by its stable identifier."""
    for template in DEFAULT_PROMPTS:
        if template.id == prompt_id:
            return template
    available = ", ".join(template.id for template in DEFAULT_PROMPTS)
    raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompt IDs: {available}")


def _coerce_fingerprint(value: ModelFingerprint | Mapping[str, Any]) -> ModelFingerprint:
    if isinstance(value, ModelFingerprint):
        return value
    required = {
        "model", "provider", "prompt", "system", "temperature", "max_tokens",
        "request_configuration", "field", "values", "distribution", "text_length", "stability", "length_source",
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
        length_source=str(value["length_source"]),
    )
