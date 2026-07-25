"""Public data models for the LLM client."""

from __future__ import annotations

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
    api_version: str = "2023-06-01"
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    extra_body: Mapping[str, Any] = field(default_factory=dict)

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
