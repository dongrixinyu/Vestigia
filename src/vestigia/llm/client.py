"""LiteLLM-backed synchronous completion client."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import litellm
from loguru import logger

from vestigia.config import NETWORK_RETRY_MAX_RETRIES, SYSTEM_PROMPT
from vestigia.llm.types import LLMConfig, LLMRequestError, LLMResponse, Message, Messages

_GENERATION_PARAMETER_NAMES = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "top_k",
        "presence_penalty",
        "frequency_penalty",
    }
)

litellm.disable_remote_model_cost_map = True


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

    def effective_system_prompt(self, system: str | None = None) -> str:
        """Return the mandatory system instruction actually sent to the model."""
        if system is None or system == SYSTEM_PROMPT:
            return SYSTEM_PROMPT
        if system.startswith(f"{SYSTEM_PROMPT}\n\n"):
            return system
        return f"{SYSTEM_PROMPT}\n\n{system}"

    def request_signature_context(
        self,
        prompt: str,
        *,
        system: str | None = None,
        request_parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return effective LiteLLM request controls without exposing secrets.

        Per-call generation overrides belong in ``request_parameters``; for
        example, ``{"temperature": 0.2, "max_tokens": 32}``.
        """
        request = self._request_options(
            [{"role": "user", "content": prompt}], system, request_parameters
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
        request_parameters: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        if not prompt:
            raise ValueError("prompt must not be empty")
        return self.complete_messages(
            [{"role": "user", "content": prompt}],
            system=system,
            request_parameters=request_parameters,
        )

    def complete_messages(
        self,
        messages: Messages,
        *,
        system: str | None = None,
        request_parameters: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        """Complete messages using optional per-call generation overrides.

        Put every override in ``request_parameters``, such as
        ``{"temperature": 0.2, "max_tokens": 32}``; individual generation
        keyword arguments are intentionally not supported.
        """
        if not messages:
            raise ValueError("messages must not be empty")
        request = self._request_options(messages, system, request_parameters)
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
        normalized = self._normalize(response)
        logger.info(
            "LLM request succeeded (model={}, endpoint={}, request_id={})",
            normalized.model,
            self.config.endpoint or self.config.base_url,
            normalized.request_id or "unknown",
        )
        return normalized

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
        request_parameters: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_messages = [dict(message) for message in messages]
        system_prompt = self.effective_system_prompt(system)
        request_messages.insert(0, {"role": "system", "content": system_prompt})
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
        self._add_generation_options(options, request_parameters)
        extra_body = dict(self.config.extra_body)
        top_k = self._effective_top_k(request_parameters)
        if top_k is not None:
            extra_body["top_k"] = top_k
        if extra_body:
            options["extra_body"] = extra_body
        return options

    def _litellm_model(self) -> str:
        prefix = "anthropic" if self.config.provider == "anthropic" else "openai"
        if self.config.model.startswith(f"{prefix}/"):
            return self.config.model
        return f"{prefix}/{self.config.model}"

    def _effective_top_k(self, request_parameters: Mapping[str, Any] | None) -> int | None:
        """Resolve top_k; LiteLLM receives it inside provider-specific extra_body."""
        overrides = _validated_generation_parameters(request_parameters)
        top_k = overrides.get("top_k", self.config.top_k)
        return int(top_k) if top_k is not None else None

    def _add_generation_options(
        self, options: dict[str, Any], request_parameters: Mapping[str, Any] | None
    ) -> None:
        overrides = _validated_generation_parameters(request_parameters)
        values = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "presence_penalty": self.config.presence_penalty,
            "frequency_penalty": self.config.frequency_penalty,
        }
        values.update({name: value for name, value in overrides.items() if name != "top_k"})
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



def _validated_generation_parameters(
    request_parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate per-call generation overrides accepted by :class:`LLMClient`."""
    if request_parameters is None:
        return {}
    unsupported = request_parameters.keys() - _GENERATION_PARAMETER_NAMES
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"unsupported request_parameters keys: {names}")
    return {
        name: dict(value) if name == "reasoning" and isinstance(value, Mapping) else value
        for name, value in request_parameters.items()
    }

def _is_network_connection_error(exc: BaseException) -> bool:
    """Recognize connection and timeout errors emitted by LiteLLM transports.

    LiteLLM can wrap exceptions from httpx, httpcore, requests, or its own
    exception classes, so inspect the exception chain rather than depending on
    one optional HTTP implementation.
    """
    network_names = {
        "APIConnectionError",
        "APIConnectionTimeoutError",
        "APITimeoutError",
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
