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
    "build_model_fingerprint",
    "compare_fingerprint_to_reference",
    "create_fingerprint",
    "identify_fingerprint",
    "load_fingerprint",
    "predict_distribution",
    "save_fingerprint",
    "text_parser",
]
__version__ = "0.1.0"
