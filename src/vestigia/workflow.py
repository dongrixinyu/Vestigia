"""One-call workflows for creating and verifying persisted model fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vestigia.config import DEFAULT_REQUEST_PARAMS, SYSTEM_PROMPT
from vestigia.identify import (
    FingerprintIdentificationResult,
    FingerprintTestResult,
    ModelFingerprint,
    Parser,
    build_model_fingerprint,
    compare_fingerprint_to_reference,
)
from vestigia.llm import LLMClient, LLMConfig
from vestigia.prompts import DEFAULT_PROMPTS, PromptTemplate
from vestigia.prompts.base import FeatureKind

_REQUEST_PARAM_NAMES = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "reasoning",
        "reasoning_effort",
        "extra_body",
        "extra_headers",
}
)


def text_parser(content: str) -> dict[str, str]:
    """Default feature parser: compare the complete response text."""
    return {"text": content}


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
    request_params: Mapping[str, Any] | None = None,
    field: str | None = None,
    count: int = 50,
) -> ModelFingerprint:
    """Sample one built-in probe repeatedly and save its response distribution.

    ``prompt_id`` selects a probe from :mod:`vestigia.prompts`; arbitrary prompt
    text is deliberately not accepted. ``variant_index`` selects one fixed
    wording from that probe, which is then used for every one of the ``count``
    calls. The probe's parser is used automatically unless explicitly
    overridden.

    ``provider`` selects only the wire protocol used by the endpoint (for
    example, an OpenAI-compatible relay); it is not persisted in the
    fingerprint identity or filename.

    ``base_url``, ``api_key`` and ``model`` are the required connection values.
    Put all model request controls in ``request_params``, for example
    ``{"temperature": 0.1, "max_tokens": 64, "top_p": 0.9}``. The saved
    JSON can be passed directly to :func:`verify_fingerprint` after loading
    with :func:`load_fingerprint`.
    """
    selected_prompt, selected_parser, feature_kind, length_field = _select_prompt(
        prompt_id=prompt_id,
        variant_index=variant_index,
    )
    selected_field = field or "parsed"
    params = _validated_request_params(request_params)
    config = LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        **params,
    )
    with LLMClient(config) as client:
        fingerprint = build_model_fingerprint(
            client,
            selected_prompt,
            selected_parser,
            count=count,
            feature_kind=feature_kind,
            field=selected_field,
        )
    if output is not None:
        save_fingerprint(fingerprint, output, prompt_id=prompt_id)
    return fingerprint


def identify_fingerprint(
    reference_directory: str | Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    provider: str = "openai_compatible",
    endpoint: str | None = None,
    request_params: Mapping[str, Any] | None = None,
    parser: Parser | None = None,
    count: int = 20,
) -> FingerprintIdentificationResult:
    """Identify one sampled model by comparing it to all saved fingerprints.

    Historical references are loaded from ``reference_directory``. They must
    share one prompt, feature definition, and request configuration so the
    candidate is sampled exactly once. The returned comparisons are sorted by
    total-variation distance; ``best_match`` is the closest accepted reference,
    or ``None`` when no reference matches.
    """
    references = _load_fingerprint_directory(reference_directory)
    template = references[0]
    for reference in references[1:]:
        # This validates common experiment conditions without making a request.
        compare_fingerprint_to_reference(template, reference)

    selected_parser = parser or _parser_for_fingerprint(template)
    params = _reference_request_params(template)
    params.update(_validated_request_params(request_params))
    config = LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        base_url=base_url,
        api_key=api_key,
        model=model,
        endpoint=endpoint,
        **params,
    )
    with LLMClient(config) as client:
        candidate = build_model_fingerprint(
            client,
            template.prompt,
            selected_parser,
            feature_kind=template.feature_kind,
            field=template.field or "parsed",
            count=count,
        )

    comparisons = tuple(
        sorted(
            (compare_fingerprint_to_reference(candidate, reference) for reference in references),
            key=lambda result: result.distances["total_variation_distance"],
        )
    )
    best_match = next((result for result in comparisons if result.matches_reference), None)
    return FingerprintIdentificationResult(
        tested_model=model, comparisons=comparisons, best_match=best_match
    )


def _load_fingerprint_directory(directory: str | Path) -> tuple[ModelFingerprint, ...]:
    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"fingerprint directory does not exist: {path}")
    references = tuple(load_fingerprint(item) for item in sorted(path.glob("*.json")))
    if not references:
        raise ValueError(f"fingerprint directory contains no JSON fingerprints: {path}")
    return references


def _validated_request_params(request_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize the unified model request controls."""
    params: dict[str, Any] = {
        key: (dict(value) if isinstance(value, Mapping) else value)
        for key, value in DEFAULT_REQUEST_PARAMS.items()
    }
    if request_params is None:
        return params
    unknown = request_params.keys() - _REQUEST_PARAM_NAMES
    if unknown:
        raise ValueError(f"unsupported request_params keys: {', '.join(sorted(unknown))}")
    params.update(request_params)
    if "extra_body" in params:
        params["extra_body"] = dict(params["extra_body"] or {})
    if "extra_headers" in params:
        params["extra_headers"] = dict(params["extra_headers"] or {})
    return params


def _reference_request_params(fingerprint: ModelFingerprint) -> dict[str, Any]:
    """Recover the reference fingerprint's request controls for verification."""
    configuration = dict(fingerprint.request_configuration)
    configuration.pop("system_prompt", None)
    return configuration


def save_fingerprint(
    fingerprint: ModelFingerprint,
    output_directory: str | Path,
    *,
    prompt_id: str | None = None,
) -> Path:
    """Persist a fingerprint under a canonical, configuration-specific filename.

    ``output_directory`` is always treated as a directory. The filename is
    ``{model}__{prompt_id}__{params_hash}__{started_at}.json`` so fingerprints produced
    under different model or request settings cannot overwrite one another. Callers saving an existing fingerprint may omit ``prompt_id``;
    in that case it is recovered from the built-in prompt catalog.
    """
    resolved_prompt_id = prompt_id or _prompt_id_for_fingerprint(fingerprint)
    params_hash = _fingerprint_parameters_hash(fingerprint)
    filename = "__".join(
        (
            _filename_component(fingerprint.model),
            _filename_component(resolved_prompt_id),
            params_hash,
            _timestamp_filename_component(fingerprint.started_at),
        )
    ) + ".json"
    path = Path(output_directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": fingerprint.model,
        "prompt_id": resolved_prompt_id,
        "prompt": fingerprint.prompt,
        "parameters_hash": params_hash,
        "request_params": dict(fingerprint.request_configuration),
        "feature_kind": fingerprint.feature_kind,
        "field": fingerprint.field,
        "values": fingerprint.values,
        "distribution": fingerprint.distribution,
        "stability": fingerprint.stability,
        "started_at": fingerprint.started_at,
        "finished_at": fingerprint.finished_at,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    return path


def _effective_request_params(fingerprint: ModelFingerprint) -> dict[str, Any]:
    """Return the complete request parameters recorded with a fingerprint."""
    return dict(fingerprint.request_configuration)


def _fingerprint_parameters_hash(fingerprint: ModelFingerprint) -> str:
    """Hash all request and feature controls apart from filename identity fields."""
    parameters = {
        "prompt": fingerprint.prompt,
        "request_params": _effective_request_params(fingerprint),
        "feature_kind": fingerprint.feature_kind,
        "field": fingerprint.field,
    }
    canonical = json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _timestamp_filename_component(timestamp: str | None) -> str:
    """Make an RFC 3339 timestamp safe for use in a filename."""
    if timestamp is None:
        return "unknown-time"
    return timestamp


def _filename_component(value: str) -> str:
    """Make one human-readable filename component safe on common filesystems."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return safe.strip("._") or "unknown"


def _prompt_id_for_fingerprint(fingerprint: ModelFingerprint) -> str:
    matching_templates = [
        template for template in DEFAULT_PROMPTS if fingerprint.prompt in template.variants
    ]
    if len(matching_templates) != 1:
        raise ValueError(
            "cannot infer prompt_id from fingerprint prompt; pass prompt_id explicitly"
        )
    return matching_templates[0].id


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
) -> tuple[str, Parser, FeatureKind, str]:
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
    return selected_prompt, template.parser, template.feature_kind, template.length_field


def _prompt_template(prompt_id: str) -> PromptTemplate:
    """Return a built-in probe by its stable identifier."""
    for template in DEFAULT_PROMPTS:
        if template.id == prompt_id:
            return template
    available = ", ".join(template.id for template in DEFAULT_PROMPTS)
    raise ValueError(f"unknown prompt_id {prompt_id!r}; available prompt IDs: {available}")


def _coerce_saved_batch_fingerprint(value: Mapping[str, Any]) -> ModelFingerprint:
    """Rebuild the runtime fingerprint object from the compact saved format."""
    required = {
        "model", "prompt", "request_params", "feature_kind", "field",
        "values", "distribution", "stability",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"fingerprint is missing fields: {', '.join(sorted(missing))}")
    raw_params = dict(value["request_params"])
    system_prompt = raw_params.get("system_prompt", SYSTEM_PROMPT)
    if not isinstance(system_prompt, str):
        raise ValueError("fingerprint request_params.system_prompt must be a string")
    params = _validated_request_params({
        key: item for key, item in raw_params.items() if key != "system_prompt"
    })
    params["system_prompt"] = system_prompt
    return ModelFingerprint(
        model=str(value["model"]),
        prompt=str(value["prompt"]),
        request_configuration=params,
        feature_kind=str(value["feature_kind"]),  # type: ignore[arg-type]
        field=str(value["field"]) if isinstance(value["field"], str) else None,
        values=tuple(str(item) for item in value["values"]),
        distribution=dict(value["distribution"]),
        stability=dict(value["stability"]),
        started_at=str(value["started_at"]) if isinstance(value.get("started_at"), str) else None,
        finished_at=str(value["finished_at"]) if isinstance(value.get("finished_at"), str) else None,
    )


def _coerce_fingerprint(value: ModelFingerprint | Mapping[str, Any]) -> ModelFingerprint:
    if isinstance(value, ModelFingerprint):
        return value
    if "request_params" in value:
        return _coerce_saved_batch_fingerprint(value)
    required = {
        "model", "prompt", "system", "temperature", "max_tokens",
        "request_configuration", "feature_kind", "field", "length_field", "values", "distribution", "stability",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"fingerprint is missing fields: {', '.join(sorted(missing))}")
    return ModelFingerprint(
        model=str(value["model"]),
        prompt=str(value["prompt"]),
        request_configuration=dict(value["request_configuration"]),
        feature_kind=str(value["feature_kind"]),  # type: ignore[arg-type]
        field=str(value["field"]) if isinstance(value["field"], str) else None,
        values=tuple(str(item) for item in value["values"]),
        distribution=dict(value["distribution"]),
        stability=dict(value["stability"]),
        started_at=str(value["started_at"]) if isinstance(value.get("started_at"), str) else None,
        finished_at=str(value["finished_at"]) if isinstance(value.get("finished_at"), str) else None,
    )
