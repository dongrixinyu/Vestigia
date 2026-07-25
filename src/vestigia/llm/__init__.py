"""Provider-agnostic, non-streaming LLM completion client."""

from vestigia.llm.client import LLMClient
from vestigia.llm.models import LLMConfig, LLMRequestError, LLMResponse

__all__ = ["LLMClient", "LLMConfig", "LLMRequestError", "LLMResponse"]
