"""One-call workflows for creating and verifying persisted model fingerprints."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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
from vestigia.validation import compare_distributions

_REQUEST_PARAM_NAMES = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
        "extra_body",
        "extra_headers",
}
)


DistanceType = str
_DISTANCE_RESULT_KEYS: dict[DistanceType, str] = {
    "total_variation": "total_variation_distance",
    "jensen_shannon": "jensen_shannon_distance",
}


@dataclass(frozen=True, slots=True)
class _ObservedDistributionScore:
    """Internal aggregate score for the selected distance type."""

    model: str
    distance: float
    feature_matches: tuple[ObservedFeatureMatch, ...]


@dataclass(frozen=True, slots=True)
class ObservedDistribution:
    """One observed feature distribution and its experiment identity.

    An empty ``params_hash`` is a wildcard that accepts every saved parameter
    configuration for the same ``prompt_id``.
    """

    prompt_id: str
    params_hash: str
    values: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "params_hash": self.params_hash,
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class ObservedFeatureMatch:
    """One reference feature selected for a model and an observed distribution."""

    prompt_id: str
    params_hash: str
    distance_type: DistanceType
    distance: float
    fingerprint_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "params_hash": self.params_hash,
            "distance_type": self.distance_type,
            "distance": self.distance,
            "fingerprint_path": str(self.fingerprint_path),
        }


@dataclass(frozen=True, slots=True)
class ObservedDistributionMatch:
    """Closest saved fingerprints and aggregate distance for one historical model."""

    model: str
    distance_type: DistanceType
    distance: float
    probability: float
    feature_matches: tuple[ObservedFeatureMatch, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "distance_type": self.distance_type,
            "distance": self.distance,
            "probability": self.probability,
            "feature_matches": [match.to_dict() for match in self.feature_matches],
        }


@dataclass(frozen=True, slots=True)
class ObservedDistributionIdentification:
    """Offline prediction result for externally collected sample values."""

    values: tuple[str, ...]
    observed_distributions: tuple[ObservedDistribution, ...]
    matches: tuple[ObservedDistributionMatch, ...]
    distance_type: DistanceType
    softmax_temperature: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": list(self.values),
            "observed_distributions": [item.to_dict() for item in self.observed_distributions],
            "distance_type": self.distance_type,
            "softmax_temperature": self.softmax_temperature,
            "matches": [match.to_dict() for match in self.matches],
        }


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
    count: int = 50,
) -> ModelFingerprint:
    """Sample one built-in probe repeatedly and save its response distribution.

    ``prompt_id`` selects a probe from :mod:`vestigia.prompts`; each built-in
    probe owns its parser and the parsed field used as its fingerprint feature.
    Arbitrary prompt text and caller-selected fields are deliberately not
    accepted. ``variant_index`` selects one fixed wording from that probe,
    which is then used for every one of the ``count`` calls.

    ``provider`` selects only the wire protocol used by the endpoint (for
    example, an OpenAI-compatible relay); it is not persisted in the
    fingerprint identity or filename.

    ``base_url``, ``api_key`` and ``model`` are the required connection values.
    Put all model request controls in ``request_params``, for example
    ``{"temperature": 0.1, "max_tokens": 64, "top_p": 0.9}`` for portable
    controls, and put endpoint-specific controls such as ``reasoning_effort``
    in ``extra_body``. The saved
    JSON can be passed directly to :func:`verify_fingerprint` after loading
    with :func:`load_fingerprint`.
    """
    selected_prompt, selected_parser, selected_field, feature_kind, length_field = _select_prompt(
        prompt_id=prompt_id,
        variant_index=variant_index,
    )
    selected_field = selected_field if feature_kind == "parsed" else "parsed"
    system = _prompt_template(prompt_id).system
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
            system=system,
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
    system = _system_for_fingerprint(template)
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
            system=system,
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


def predict_distribution(
    observed_distributions: (
        list[ObservedDistribution | Mapping[str, Any]]
        | tuple[ObservedDistribution | Mapping[str, Any], ...]
    ),
    reference_directory: str | Path,
    *,
    distance_type: DistanceType = "total_variation",
    softmax_temperature: float = 0.1,
) -> ObservedDistributionIdentification:
    """Identify models by aggregating multiple experiment-matched features.

    Every observed item must contain ``prompt_id``, ``params_hash``, and
    ``values``. A non-empty ``params_hash`` must match exactly; an empty one
    matches every saved parameter configuration for that prompt. Each model
    must cover every supplied feature; its final distance is the equal-weight
    mean of its per-feature distances.
    """
    if distance_type not in _DISTANCE_RESULT_KEYS:
        supported = ", ".join(sorted(_DISTANCE_RESULT_KEYS))
        raise ValueError(f"unsupported distance_type {distance_type!r}; expected one of: {supported}")
    distance_result_key = _DISTANCE_RESULT_KEYS[distance_type]
    observed = tuple(_coerce_observed_distribution(item) for item in observed_distributions)
    if not observed:
        raise ValueError("observed_distributions must not be empty")
    identities = [(item.prompt_id, item.params_hash) for item in observed]
    if len(set(identities)) != len(identities):
        raise ValueError("each observed distribution must have a unique prompt_id and params_hash")
    if softmax_temperature <= 0:
        raise ValueError("softmax_temperature must be greater than zero")

    directory = Path(reference_directory)
    paths = tuple(sorted(directory.rglob("*.json"))) if directory.is_dir() else ()
    if not paths:
        raise ValueError(f"fingerprint directory contains no JSON fingerprints: {directory}")

    references: dict[tuple[str, str], dict[str, list[tuple[ModelFingerprint, Path, str]]]] = {}
    for path in paths:
        payload = json.loads(path.read_text("utf-8"))
        if not isinstance(payload, Mapping):
            continue
        prompt_id = payload.get("prompt_id")
        params_hash = payload.get("parameters_hash")
        if not isinstance(prompt_id, str) or not isinstance(params_hash, str):
            continue
        matching_identities = [
            identity
            for identity in identities
            if identity[0] == prompt_id and (not identity[1] or identity[1] == params_hash)
        ]
        if not matching_identities:
            continue
        reference = load_fingerprint(path)
        for identity in matching_identities:
            references.setdefault(identity, {}).setdefault(reference.model, []).append(
                (reference, path, params_hash)
            )

    missing = [identity for identity in identities if identity not in references]
    if missing:
        formatted = ", ".join(f"{prompt_id}/{params_hash}" for prompt_id, params_hash in missing)
        raise ValueError(f"no saved fingerprints match observed distributions: {formatted}")

    common_models = set.intersection(*(set(references[identity]) for identity in identities))
    if not common_models:
        raise ValueError("no model has saved fingerprints for every observed distribution")

    scored: list[_ObservedDistributionScore] = []
    for model in sorted(common_models):
        feature_matches: list[ObservedFeatureMatch] = []
        selected_distances: list[float] = []
        for item in observed:
            candidates = []
            for reference, path, reference_params_hash in references[(item.prompt_id, item.params_hash)][model]:
                distances = compare_distributions(reference.values, item.values)
                candidates.append((distances, path, reference_params_hash))
            distances, path, reference_params_hash = min(
                candidates, key=lambda candidate: candidate[0][distance_result_key]
            )
            selected_distances.append(distances[distance_result_key])
            feature_matches.append(
                ObservedFeatureMatch(
                    prompt_id=item.prompt_id,
                    params_hash=reference_params_hash,
                    distance_type=distance_type,
                    distance=distances[distance_result_key],
                    fingerprint_path=path,
                )
            )
        scored.append(
            _ObservedDistributionScore(
                model=model,
                distance=sum(selected_distances) / len(selected_distances),
                feature_matches=tuple(feature_matches),
            )
        )

    scored.sort(key=lambda item: item.distance)
    logits = [-(item.distance / softmax_temperature) for item in scored]
    maximum = max(logits)
    weights = [math.exp(logit - maximum) for logit in logits]
    normalizer = sum(weights)
    matches = tuple(
        ObservedDistributionMatch(
            model=item.model,
            distance_type=distance_type,
            distance=item.distance,
            probability=weight / normalizer,
            feature_matches=item.feature_matches,
        )
        for item, weight in zip(scored, weights, strict=True)
    )
    return ObservedDistributionIdentification(
        values=tuple(value for item in observed for value in item.values),
        observed_distributions=observed,
        matches=matches,
        distance_type=distance_type,
        softmax_temperature=softmax_temperature,
    )


def _coerce_observed_distribution(
    value: ObservedDistribution | Mapping[str, Any],
) -> ObservedDistribution:
    if isinstance(value, ObservedDistribution):
        return value
    required = {"prompt_id", "params_hash", "values"}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"observed distribution is missing fields: {', '.join(sorted(missing))}")
    prompt_id = value["prompt_id"]
    params_hash = value["params_hash"]
    values = value["values"]
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ValueError("observed distribution prompt_id must be a non-empty string")
    if not isinstance(params_hash, str):
        raise ValueError("observed distribution params_hash must be a string")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("observed distribution values must be a non-empty list or tuple")
    return ObservedDistribution(prompt_id, params_hash, tuple(str(item) for item in values))


def _load_fingerprint_directory(directory: str | Path) -> tuple[ModelFingerprint, ...]:
    """Load all persisted fingerprints from an existing directory."""
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


def _system_for_fingerprint(fingerprint: ModelFingerprint) -> str | None:
    """Recover a built-in probe's system instruction for a saved fingerprint."""
    matching_templates = [
        template for template in DEFAULT_PROMPTS if fingerprint.prompt in template.variants
    ]
    return matching_templates[0].system if len(matching_templates) == 1 else None

def _select_prompt(
    *,
    prompt_id: str,
    variant_index: int,
) -> tuple[str, Parser, str, FeatureKind, str]:
    """Resolve one fixed wording, parser, and feature field from a built-in probe."""
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
    return (
        selected_prompt,
        template.parser,
        template.field,
        template.feature_kind,
        template.length_field,
    )


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
    # Fingerprints saved before provider-specific controls were moved to
    # ``extra_body`` may contain these deprecated top-level fields. Preserve
    # their behavior by migrating them while loading the historical JSON.
    extra_body = dict(raw_params.get("extra_body") or {})
    for name in ("top_k", "reasoning", "reasoning_effort"):
        deprecated_value = raw_params.pop(name, None)
        if deprecated_value is not None and name not in extra_body:
            extra_body[name] = deprecated_value
    raw_params["extra_body"] = extra_body
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
