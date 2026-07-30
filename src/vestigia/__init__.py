"""Vestigia: tools for collecting and identifying LLM behavioral fingerprints."""

from vestigia.identify import (
    FingerprintIdentificationResult,
    FingerprintTestResult,
    ModelFingerprint,
    build_model_fingerprint,
    compare_fingerprint_to_reference,
)
from vestigia.llm import LLMClient, LLMConfig, LLMRequestError, LLMResponse
from vestigia.prompts import DEFAULT_PROMPTS, PromptTemplate
from vestigia.request_params import (
    DEFAULT_REQUEST_PARAM_PRESET,
    REQUEST_PARAM_PRESETS,
    available_request_param_presets,
    get_request_params,
)
from vestigia.workflow import (
    ObservedDistributionIdentification,
    ObservedDistribution,
    ObservedDistributionMatch,
    ObservedFeatureMatch,
    create_fingerprint,
    identify_fingerprint,
    load_fingerprint,
    predict_distribution,
    save_fingerprint,
    text_parser,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "DEFAULT_REQUEST_PARAM_PRESET",
    "FingerprintIdentificationResult",
    "FingerprintTestResult",
    "LLMClient",
    "LLMConfig",
    "LLMRequestError",
    "LLMResponse",
    "ModelFingerprint",
    "ObservedDistributionIdentification",
    "ObservedDistribution",
    "ObservedDistributionMatch",
    "ObservedFeatureMatch",
    "PromptTemplate",
    "REQUEST_PARAM_PRESETS",
    "build_model_fingerprint",
    "available_request_param_presets",
    "compare_fingerprint_to_reference",
    "create_fingerprint",
    "identify_fingerprint",
    "get_request_params",
    "load_fingerprint",
    "predict_distribution",
    "save_fingerprint",
    "text_parser",
]
__version__ = "0.1.0"
