"""Versioned, provider-neutral request-parameter presets for fingerprinting.

Use :func:`get_request_params` to obtain a mutable copy for an API call.  Do
not change a preset in place: changing sampling controls creates a different
behavioral experiment and therefore a different fingerprint.
"""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import Any, Final, Mapping

RequestParams = Mapping[str, Any]

# This profile uses the project's portable request controls plus the fixed
# reasoning controls required by the fingerprint experiment. Endpoints that do
# not support them should use a separately versioned compatible preset.
_STANDARD_V1: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "temperature": 0.1,
        "max_tokens": None,
        "top_p": 1.0,
        "top_k": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": MappingProxyType(
            {
                "reasoning": True,
                "reasoning_effort": "low",
            }
        ),
        "extra_headers": MappingProxyType({}),
    }
)

# Lower temperature is useful when a deployment needs less generation noise,
# but it is a distinct experiment and must never be mixed with standard-v1.
_LOW_VARIANCE_V1: Final[Mapping[str, Any]] = MappingProxyType(
    {
        **_STANDARD_V1,
        "temperature": 0.1,
    }
)

REQUEST_PARAM_PRESETS: Final[Mapping[str, RequestParams]] = MappingProxyType(
    {
        "fingerprint_standard_v1": _STANDARD_V1,
        "fingerprint_low_variance_v1": _LOW_VARIANCE_V1,
    }
)
"""Immutable catalog of named, versioned request-parameter presets."""

DEFAULT_REQUEST_PARAM_PRESET: Final[str] = "fingerprint_standard_v1"
"""Default portable fingerprint experiment profile."""


def get_request_params(name: str = DEFAULT_REQUEST_PARAM_PRESET) -> dict[str, Any]:
    """Return an independent copy of a named request-parameter preset.

    The returned dictionary is safe to pass to ``create_fingerprint``. Its
    nested mappings are copied too, so a caller cannot mutate the catalog
    shared by later requests. Use :func:`available_request_param_presets` to
    discover valid names.
    """
    try:
        preset = REQUEST_PARAM_PRESETS[name]
    except KeyError as error:
        supported = ", ".join(sorted(REQUEST_PARAM_PRESETS))
        raise ValueError(f"unknown request-parameter preset {name!r}; expected one of: {supported}") from error
    return _mutable_copy(preset)


def _mutable_copy(value: Any) -> Any:
    """Copy immutable catalog values into ordinary mutable Python containers."""
    if isinstance(value, Mapping):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_copy(item) for item in value]
    return deepcopy(value)


def available_request_param_presets() -> tuple[str, ...]:
    """Return all stable preset names in deterministic order."""
    return tuple(sorted(REQUEST_PARAM_PRESETS))


__all__ = [
    "DEFAULT_REQUEST_PARAM_PRESET",
    "REQUEST_PARAM_PRESETS",
    "RequestParams",
    "available_request_param_presets",
    "get_request_params",
]
