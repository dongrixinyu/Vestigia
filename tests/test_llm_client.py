from __future__ import annotations

from unittest.mock import patch

import pytest

from vestigia import LLMClient, LLMConfig, LLMRequestError


@patch("vestigia.llm.client.litellm.completion")
def test_client_routes_openai_compatible_request_through_litellm(completion) -> None:
    completion.return_value = {
        "id": "chatcmpl_123",
        "model": "gateway-model",
        "choices": [{"message": {"content": "final answer"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2},
    }
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example/v1",
            api_key="secret",
            model="configured-model",
            temperature=0.2,
        )
    )

    response = client.complete("hello", system="be concise", max_tokens=32)

    completion.assert_called_once_with(
        model="openai/configured-model",
        messages=[
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        api_key="secret",
        api_base="https://gateway.example/v1",
        timeout=60.0,
        stream=False,
        num_retries=0,
        temperature=0.2,
        max_tokens=32,
    )
    assert response.text == "final answer"
    assert response.reasoning_content is None
    assert response.model == "gateway-model"
    assert response.request_id == "chatcmpl_123"


@patch("vestigia.llm.client.litellm.completion")
def test_client_preserves_litellm_reasoning_content_separately_from_final_text(completion) -> None:
    completion.return_value = {
        "choices": [{
            "message": {"content": "42", "reasoning_content": "reasoning trace"},
            "finish_reason": "stop",
        }],
    }
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example/v1",
            api_key="secret",
            model="reasoning-model",
        )
    )

    response = client.complete("hello")

    assert response.text == "42"
    assert response.reasoning_content == "reasoning trace"

    completion.return_value = {
        "id": "msg_123",
        "model": "claude-test",
        "choices": [{"message": {"content": "hello"}, "finish_reason": "end_turn"}],
    }
    client = LLMClient(
        LLMConfig(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="secret",
            model="claude-test",
            max_tokens=100,
        )
    )

    assert client.complete("hello").text == "hello"
    assert completion.call_args.kwargs["model"] == "anthropic/claude-test"
    assert completion.call_args.kwargs["api_version"] == "2023-06-01"


def test_generation_controls_are_exposed_in_litellm_signature_context() -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example/v1",
            api_key="secret",
            model="configured-model",
            temperature=0.7,
            max_tokens=1024,
            top_p=1.0,
            top_k=40,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            reasoning={"effort": "high"},
            reasoning_effort="high",
        )
    )

    context = client.request_signature_context("hello")

    assert context["request_model"] == "openai/configured-model"
    assert context["generation_parameters"] == {
        "stream": False,
        "num_retries": 0,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1.0,
        "top_k": 40,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.3,
        "reasoning": {"effort": "high"},
        "reasoning_effort": "high",
    }


def test_config_rejects_multi_completion_and_stream_overrides() -> None:
    with pytest.raises(ValueError, match="n, stream"):
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example",
            api_key="secret",
            model="model",
            extra_body={"n": 2, "stream": True},
        )


@patch("vestigia.llm.client.litellm.completion", side_effect=RuntimeError("bad key"))
def test_litellm_errors_are_normalized(_completion) -> None:
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example",
            api_key="bad",
            model="model",
        )
    )

    with pytest.raises(LLMRequestError, match="LiteLLM request failed: bad key"):
        client.complete("hello")
