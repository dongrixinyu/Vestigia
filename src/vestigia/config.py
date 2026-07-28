"""Application-wide runtime settings."""

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
