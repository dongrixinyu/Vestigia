"""Application-wide runtime settings."""

from typing import Any

# Complete model-request parameter set. These values are sent to the endpoint
# and recorded in every fingerprint. ``None`` means the parameter is omitted
# because it has no portable value across providers.
#
# Provider- or gateway-specific controls (for example ``reasoning``,
# ``reasoning_effort``, ``seed`` or ``cache``) belong in ``extra_body``.
# ``top_k`` is configured uniformly here, but is nested into ``extra_body``
# at dispatch because LiteLLM routes it differently across providers.
DEFAULT_REQUEST_PARAMS: dict[str, Any] = {
    "temperature": 1.0,
    "max_tokens": None,
    "top_p": 1.0,
    # None means omit top_k; explicit 0 has provider-specific semantics.
    "top_k": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "extra_body": {},
    "extra_headers": {},
}

# Standard instruction prepended to every LLM request. Keep this value stable
# while collecting and comparing fingerprints; changing it changes the model's
# behavior and therefore produces a different fingerprint.
SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's instructions exactly."

# Stability validation settings used after collecting a fingerprint distribution.
# They are intentionally global internal settings; create_fingerprint exposes
# only ``count`` as the model sampling count.
STABILITY_SUBSET_SIZE = 20
STABILITY_RESAMPLES = 1_000
STABILITY_SEED = 0
MAX_P95_TV_DISTANCE = 0.20

# Number of retries after the initial request when a network connection error
# occurs. Change this value to tune retry behavior for all LLM requests.
NETWORK_RETRY_MAX_RETRIES = 5
