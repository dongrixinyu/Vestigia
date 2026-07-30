from __future__ import annotations

import pytest

from vestigia.request_params import (
    DEFAULT_REQUEST_PARAM_PRESET,
    REQUEST_PARAM_PRESETS,
    available_request_param_presets,
    get_request_params,
)


def test_standard_request_params_are_complete() -> None:
    params = get_request_params()

    assert DEFAULT_REQUEST_PARAM_PRESET == "fingerprint_standard_v1"
    assert params == {
        "temperature": 0.1,
        "max_tokens": None,
        "top_p": 1.0,
        "top_k": None,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "extra_body": {
            "reasoning": True,
            "reasoning_effort": "low",
        },
        "extra_headers": {},
    }
    assert set(available_request_param_presets()) == set(REQUEST_PARAM_PRESETS)


def test_request_param_preset_returns_an_independent_mutable_copy() -> None:
    first = get_request_params("fingerprint_low_variance_v1")
    first["temperature"] = 0.9
    first["extra_body"]["reasoning"] = True

    second = get_request_params("fingerprint_low_variance_v1")

    assert second["temperature"] == 0.1
    assert second["extra_body"] == {
        "reasoning": True,
        "reasoning_effort": "low",
    }


def test_request_param_preset_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown request-parameter preset"):
        get_request_params("missing")
