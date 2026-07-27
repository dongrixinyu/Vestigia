"""Synchronous adapters for supported LLM HTTP APIs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import uuid4

import httpx

from vestigia.llm.models import LLMConfig, LLMRequestError, LLMResponse, Message, Messages


class LLMClient:
    """Call an LLM endpoint and return its final, non-streaming response.

    The client owns an :class:`httpx.Client` unless one is supplied. Supplying a
    client is useful for application-wide connection management and for tests;
    caller-supplied clients are never closed by :meth:`close`.
    """

    def __init__(self, config: LLMConfig, *, http_client: httpx.Client | None = None) -> None:
        self.config = config
        self._http_client = http_client or httpx.Client(timeout=config.timeout)
        self._owns_http_client = http_client is None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the internally-created HTTP client, if any."""
        if self._owns_http_client:
            self._http_client.close()

    def request_signature_context(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Return the effective, non-secret request fields used in a signature.

        The API key and values of user-supplied headers are represented by
        SHA-256 digests, so JSONL signature records remain safe to share.
        """
        messages: list[Message] = [{"role": "user", "content": prompt}]
        if self.config.provider == "openai_compatible":
            payload = self._openai_payload(messages, system, temperature, max_tokens)
        elif self.config.provider == "anthropic":
            payload = self._anthropic_payload(messages, system, temperature, max_tokens)
        else:
            raise ValueError(f"unsupported provider: {self.config.provider!r}")
        return {
            "request_model": str(payload["model"]),
            "api_key_sha256": _sha256(self.config.api_key),
            "request_url": self._endpoint(
                "chat/completions" if self.config.provider == "openai_compatible" else "messages"
            ),
            "api_version": (
                self.config.api_version if self.config.provider == "anthropic" else None
            ),
            "extra_headers": {
                name: _sha256(value) for name, value in sorted(self.config.extra_headers.items())
            },
            "generation_parameters": {
                name: value
                for name, value in payload.items()
                if name not in {"model", "messages", "system", "stream"}
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
        """Submit one user prompt and return the completed response."""
        if not prompt:
            raise ValueError("prompt must not be empty")
        messages: list[Message] = [{"role": "user", "content": prompt}]
        return self.complete_messages(
            messages,
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
        """Submit conversation messages in the common ``role``/``content`` format."""
        if not messages:
            raise ValueError("messages must not be empty")

        if self.config.provider == "openai_compatible":
            payload = self._openai_payload(messages, system, temperature, max_tokens)
            return self._send_openai(payload)
        if self.config.provider == "anthropic":
            payload = self._anthropic_payload(messages, system, temperature, max_tokens)
            return self._send_anthropic(payload)
        # Retained for a clearer failure if a config is constructed without type checking.
        raise ValueError(f"unsupported provider: {self.config.provider!r}")

    def _openai_payload(
        self,
        messages: Messages,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        request_messages = [dict(message) for message in messages]
        if system is not None:
            request_messages.insert(0, {"role": "system", "content": system})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request_messages,
            "stream": False,
        }
        self._set_generation_options(payload, temperature, max_tokens, max_tokens_key="max_tokens")
        payload.update(self.config.extra_body)
        return payload

    def _anthropic_payload(
        self,
        messages: Messages,
        system: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        request_messages: list[dict[str, Any]] = []
        system_parts: list[str] = [system] if system is not None else []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                else:
                    raise ValueError("Anthropic system message content must be a string")
            elif role in {"user", "assistant"}:
                request_messages.append(dict(message))
            else:
                raise ValueError(f"unsupported Anthropic message role: {role!r}")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request_messages,
            "max_tokens": max_tokens or self.config.max_tokens or 1024,
            "stream": False,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        actual_temperature = temperature if temperature is not None else self.config.temperature
        if actual_temperature is not None:
            payload["temperature"] = actual_temperature
        payload.update(self.config.extra_body)
        return payload

    def _set_generation_options(
        self,
        payload: dict[str, Any],
        temperature: float | None,
        max_tokens: int | None,
        *,
        max_tokens_key: str,
    ) -> None:
        actual_temperature = temperature if temperature is not None else self.config.temperature
        actual_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        if actual_temperature is not None:
            payload["temperature"] = actual_temperature
        if actual_max_tokens is not None:
            payload[max_tokens_key] = actual_max_tokens

    def _send_openai(self, payload: Mapping[str, Any]) -> LLMResponse:
        response = self._post(
            self._endpoint("chat/completions"),
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            payload=payload,
        )
        data = self._json(response)
        try:
            choice = data["choices"][0]
            text = _content_to_text(choice["message"]["content"])
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise LLMRequestError("invalid OpenAI-compatible response format") from exc
        return LLMResponse(
            text=text,
            model=str(data.get("model", self.config.model)),
            provider="openai_compatible",
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
            request_id=response.headers.get("x-request-id") or data.get("id"),
            raw=data,
        )

    def _send_anthropic(self, payload: Mapping[str, Any]) -> LLMResponse:
        response = self._post(
            self._endpoint("messages"),
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": self.config.api_version,
            },
            payload=payload,
        )
        data = self._json(response)
        try:
            text = "".join(
                str(block["text"]) for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LLMRequestError("invalid Anthropic response format") from exc
        if not text and not data.get("content"):
            raise LLMRequestError("Anthropic response contains no content")
        return LLMResponse(
            text=text,
            model=str(data.get("model", self.config.model)),
            provider="anthropic",
            finish_reason=data.get("stop_reason"),
            usage=data.get("usage"),
            request_id=response.headers.get("request-id") or data.get("id"),
            raw=data,
        )

    def _endpoint(self, resource: str) -> str:
        if self.config.endpoint:
            return self.config.endpoint
        base_url = self.config.base_url.rstrip("/")
        if self.config.provider == "anthropic":
            return f"{base_url}/messages" if base_url.endswith("/v1") else f"{base_url}/v1/messages"
        return f"{base_url}/{resource}"

    def _post(
        self, url: str, *, headers: Mapping[str, str], payload: Mapping[str, Any]
    ) -> httpx.Response:
        cache_headers = (
            {"Cache-Control": "no-cache, no-store, max-age=0", "Pragma": "no-cache"}
            if self.config.disable_response_cache
            else {}
        )
        merged_headers = {**headers, **cache_headers, **self.config.extra_headers}
        if self.config.cache_bust_query_param:
            url = str(
                httpx.URL(url).copy_add_param(self.config.cache_bust_query_param, uuid4().hex)
            )
        try:
            response = self._http_client.post(url, headers=merged_headers, json=payload)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise LLMRequestError(
                f"LLM request failed with HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
                response_body=exc.response.text,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMRequestError(f"LLM request failed: {exc}") from exc

    @staticmethod
    def _json(response: httpx.Response) -> Mapping[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMRequestError(
                "LLM endpoint returned invalid JSON",
                status_code=response.status_code,
                response_body=response.text,
            ) from exc
        if not isinstance(data, Mapping):
            raise LLMRequestError("LLM endpoint returned a non-object JSON response")
        return data



def _sha256(value: str) -> str:
    """Hash a sensitive request value before it enters a persisted signature."""
    return hashlib.sha256(value.encode()).hexdigest()


def _content_to_text(content: Any) -> str:
    """Normalize plain and content-part response formats used by compatible APIs."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return "".join(
            str(part["text"])
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") in {"text", "output_text"}
            and "text" in part
        )
    raise ValueError("response message content is not text")
