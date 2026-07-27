"""Public data models for the LLM client."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

Provider = Literal["openai_compatible", "anthropic"]
Message = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Connection and generation defaults for one LLM endpoint.

    ``base_url`` accepts both a host URL and a versioned API root (for example,
    ``https://api.example.com/v1``). Set ``endpoint`` when a gateway uses a
    non-standard route.
    """

    provider: Provider
    base_url: str
    api_key: str
    model: str
    endpoint: str | None = None
    timeout: float = 60.0
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    reasoning: Mapping[str, Any] | None = None
    reasoning_effort: str | None = None
    api_version: str = "2023-06-01"
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    disable_response_cache: bool = True
    cache_bust_query_param: str | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must not be empty")
        if not self.api_key.strip():
            raise ValueError("api_key must not be empty")
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.top_p is not None and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between zero and one")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self.reasoning is not None and not isinstance(self.reasoning, Mapping):
            raise ValueError("reasoning must be a mapping")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("reasoning_effort must not be empty")
        forbidden_body_fields = {"n", "stream"} & self.extra_body.keys()
        if forbidden_body_fields:
            raise ValueError(
                "extra_body must not set " + ", ".join(sorted(forbidden_body_fields))
                + "; Vestigia always makes one non-streaming completion"
            )
        if self.cache_bust_query_param is not None and not self.cache_bust_query_param.strip():
            raise ValueError("cache_bust_query_param must not be empty")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized final result from an LLM request."""

    text: str
    model: str
    provider: Provider
    finish_reason: str | None
    usage: Mapping[str, Any] | None
    request_id: str | None
    raw: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RequestSignature:
    """Deterministic fingerprint of every parameter that influences an LLM response.

    All fields that can change the output — model name, generation knobs,
    system prompt, user prompt, and any extra body parameters — are folded
    into a canonical JSON string and hashed with SHA-256.  Two calls that
    produce identical ``RequestSignature`` objects are guaranteed to have
    sent byte-for-byte the same logical request; any difference in inputs
    yields a different ``digest``.

    Usage::

        sig = RequestSignature(
            model="gpt-4o",
            provider="openai_compatible",
            temperature=0.7,
            max_tokens=256,
            system="You are helpful.",
            prompt="What is 2+2?",
        )
        record["request_signature"] = sig.digest()
    """

    model: str
    provider: Provider
    prompt: str
    prompt_id: str
    api_key_sha256: str | None = None
    request_url: str | None = None
    api_version: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    generation_parameters: Mapping[str, Any] = field(default_factory=dict)
    temperature: float | None = None
    max_tokens: int | None = None
    system: str | None = None
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    disable_response_cache: bool = True
    cache_bust_query_param: str | None = None

    def _canonical(self) -> str:
        """Stable, sorted JSON representation of all signature fields."""
        obj: dict[str, Any] = {
            "model": self.model,
            "provider": self.provider,
            "prompt": self.prompt,
            "prompt_id": self.prompt_id,
        }
        if self.api_key_sha256 is not None:
            obj["api_key_sha256"] = self.api_key_sha256
        if self.request_url is not None:
            obj["request_url"] = self.request_url
        if self.api_version is not None:
            obj["api_version"] = self.api_version
        if self.extra_headers:
            obj["extra_headers"] = self.extra_headers
        if self.generation_parameters:
            obj["generation_parameters"] = self.generation_parameters
        if self.temperature is not None:
            obj["temperature"] = self.temperature
        if self.max_tokens is not None:
            obj["max_tokens"] = self.max_tokens
        if self.system is not None:
            obj["system"] = self.system
        if self.extra_body:
            obj["extra_body"] = self.extra_body
        obj["disable_response_cache"] = self.disable_response_cache
        if self.cache_bust_query_param is not None:
            obj["cache_bust_query_param"] = self.cache_bust_query_param
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)

    def digest(self) -> str:
        """Return a hex SHA-256 digest that uniquely identifies this request configuration."""
        return hashlib.sha256(self._canonical().encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialisable representation suitable for embedding in a JSONL record."""
        return {
            "digest": self.digest(),
            "model": self.model,
            "provider": self.provider,
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "api_key_sha256": self.api_key_sha256,
            "request_url": self.request_url,
            "api_version": self.api_version,
            "extra_headers": dict(self.extra_headers) if self.extra_headers else {},
            "generation_parameters": (
                dict(self.generation_parameters) if self.generation_parameters else {}
            ),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "extra_body": dict(self.extra_body) if self.extra_body else {},
            "disable_response_cache": self.disable_response_cache,
            "cache_bust_query_param": self.cache_bust_query_param,
        }


class LLMRequestError(RuntimeError):
    """Raised when an LLM endpoint cannot return a valid final response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


Messages = Sequence[Message]
