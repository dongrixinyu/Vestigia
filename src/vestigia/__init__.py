"""Vestigia: tools for collecting and identifying LLM behavioral fingerprints."""

from vestigia.identify import (
    FingerprintTestResult,
    ModelFingerprint,
    build_model_fingerprint,
    test_model_against_fingerprint,
)
from vestigia.llm import LLMClient, LLMConfig, LLMRequestError, LLMResponse
from vestigia.prompts import DEFAULT_PROMPTS, PromptTemplate

__all__ = [
    "DEFAULT_PROMPTS",
    "FingerprintTestResult",
    "LLMClient",
    "LLMConfig",
    "LLMRequestError",
    "LLMResponse",
    "ModelFingerprint",
    "PromptTemplate",
    "build_model_fingerprint",
    "test_model_against_fingerprint",
]
__version__ = "0.1.0"
