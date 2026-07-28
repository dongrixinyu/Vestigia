"""LiteLLM-backed synchronous completion client."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import litellm
from loguru import logger

from vestigia.config import NETWORK_RETRY_MAX_RETRIES
from vestigia.llm.types import LLMConfig, LLMRequestError, LLMResponse, Message, Messages


class LLMClient:
    """Call every supported model through :func:`litellm.completion`.

    Vestigia deliberately has no provider-specific HTTP implementation. The
    legacy ``provider`` configuration only selects LiteLLM's OpenAI-compatible
    or Anthropic adapter; all connection, generation and response handling is
    performed by LiteLLM.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Compatibility no-op; LiteLLM owns request resources."""

    def request_signature_context(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return effective LiteLLM request controls without exposing secrets."""
        request = self._request_options(
            [{"role": "user", "content": prompt}], system, temperature, max_tokens
        )
        return {
            "request_model": str(request["model"]),
            "api_key_sha256": _sha256(self.config.api_key),
            "request_url": self.config.endpoint or self.config.base_url,
            "api_version": self.config.api_version if self.config.provider == "anthropic" else None,
            "extra_headers": {
                name: _sha256(value) for name, value in sorted(self.config.extra_headers.items())
            },
            "generation_parameters": {
                name: value
                for name, value in request.items()
                if name
                not in {
                    "model",
                    "messages",
                    "api_key",
                    "api_base",
                    "api_version",
                    "custom_llm_provider",
                    "extra_headers",
                    "timeout",
                }
            },
        }

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not prompt:
            raise ValueError("prompt must not be empty")
        return self.complete_messages(
            [{"role": "user", "content": prompt}],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def complete_messages(
        self,
        messages: Messages,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not messages:
            raise ValueError("messages must not be empty")
        request = self._request_options(messages, system, temperature, max_tokens)
        try:
            response = self._completion_with_network_retries(request)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            response_body = getattr(exc, "response", None)
            raise LLMRequestError(
                f"LiteLLM request failed: {exc}",
                status_code=status_code if isinstance(status_code, int) else None,
                response_body=str(response_body) if response_body is not None else None,
            ) from exc
        return self._normalize(response)

    def _completion_with_network_retries(self, request: Mapping[str, Any]) -> Any:
        """Call LiteLLM, retrying only transient network connection failures."""
        for retry_number in range(NETWORK_RETRY_MAX_RETRIES + 1):
            try:
                return litellm.completion(**request)
            except Exception as exc:
                if not _is_network_connection_error(exc):
                    raise
                if retry_number == NETWORK_RETRY_MAX_RETRIES:
                    logger.error(
                        "LLM network request failed after {} retries (model={}, endpoint={}): {}",
                        NETWORK_RETRY_MAX_RETRIES,
                        self.config.model,
                        self.config.endpoint or self.config.base_url,
                        exc,
                    )
                    raise
                logger.warning(
                    "LLM network request failed; retrying ({}/{}, model={}): {}",
                    retry_number + 1,
                    NETWORK_RETRY_MAX_RETRIES,
                    self.config.model,
                    exc,
                )
        raise AssertionError("unreachable")

    def _request_options(
        self,
        messages: Messages,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        request_messages = [dict(message) for message in messages]
        if system is not None:
            request_messages.insert(0, {"role": "system", "content": system})
        options: dict[str, Any] = {
            "model": self._litellm_model(),
            "messages": request_messages,
            "api_key": self.config.api_key,
            "api_base": self.config.endpoint or self.config.base_url,
            "timeout": self.config.timeout,
            "stream": False,
            "num_retries": 0,
        }
        if self.config.provider == "anthropic":
            options["api_version"] = self.config.api_version
        if self.config.extra_headers:
            options["extra_headers"] = dict(self.config.extra_headers)
        self._add_generation_options(options, temperature, max_tokens)
        # Config validation disallows n and stream, so output cardinality and
        # transport mode cannot be changed through passthrough fields.
        options.update(self.config.extra_body)
        return options

    def _litellm_model(self) -> str:
        prefix = "anthropic" if self.config.provider == "anthropic" else "openai"
        if self.config.model.startswith(f"{prefix}/"):
            return self.config.model
        return f"{prefix}/{self.config.model}"

    def _add_generation_options(
        self, options: dict[str, Any], temperature: float | None, max_tokens: int | None
    ) -> None:
        values = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
            "reasoning": dict(self.config.reasoning) if self.config.reasoning is not None else None,
            "reasoning_effort": self.config.reasoning_effort,
        }
        options.update({name: value for name, value in values.items() if value is not None})

    def _normalize(self, response: Any) -> LLMResponse:
        raw = _as_mapping(response)
        try:
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content")
            if not isinstance(content, str):
                raise ValueError("response content is not text")
            if reasoning_content is not None and not isinstance(reasoning_content, str):
                raise ValueError("response reasoning_content is not text")
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise LLMRequestError("LiteLLM returned an invalid completion response") from exc
        return LLMResponse(
            content=content,
            model=str(raw.get("model", self.config.model)),
            provider=self.config.provider,
            finish_reason=choice.get("finish_reason"),
            usage=raw.get("usage"),
            request_id=raw.get("id") or raw.get("_hidden_params", {}).get("request_id"),
            raw=raw,
            reasoning_content=reasoning_content,
        )


def _is_network_connection_error(exc: BaseException) -> bool:
    """Recognize connection and timeout errors emitted by LiteLLM transports.

    LiteLLM can wrap exceptions from httpx, httpcore, requests, or its own
    exception classes, so inspect the exception chain rather than depending on
    one optional HTTP implementation.
    """
    network_names = {
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "Timeout",
        "WriteError",
        "WriteTimeout",
    }
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (ConnectionError, TimeoutError, OSError)):
            return True
        if type(current).__name__ in network_names:
            return True
        current = current.__cause__ or current.__context__
    return False


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, Mapping):
            return dumped
    raise LLMRequestError("LiteLLM returned a non-mapping completion response")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
