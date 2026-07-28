"""Application-wide runtime settings."""

from typing import Any

# Complete model-request parameter set. These values are sent to the endpoint
# and recorded in every fingerprint. ``None`` means the parameter is omitted
# because it has no portable value across providers.
#
# ``reasoning`` supports three forms because provider adapters expose different
# APIs: ``None`` leaves reasoning mode untouched; ``bool`` explicitly enables
# or disables it; a mapping passes provider-specific reasoning options (for
# example a budget or effort object). Keep it ``None`` by default: forcing
# reasoning on changes behavior/cost and is unsupported by many endpoints.
# ``reasoning_effort`` is also omitted by default. Although ``high`` is a
# common semantic default for reasoning-capable APIs, OpenAI-compatible
# gateways such as DeepSeek may reject the field outright. Set it explicitly
# only for an endpoint confirmed to support it.
DEFAULT_REQUEST_PARAMS: dict[str, Any] = {
    "temperature": 1.0,
    "max_tokens": None,
    "top_p": 1.0,
    "top_k": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "reasoning": None,
    "reasoning_effort": None,
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
