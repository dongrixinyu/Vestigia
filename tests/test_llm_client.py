from __future__ import annotations

import json

import httpx
import pytest
import respx

from vestigia import LLMClient, LLMConfig, LLMRequestError


@respx.mock
def test_openai_compatible_completion_normalizes_response() -> None:
    route = respx.post("https://gateway.example/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"x-request-id": "req_123"},
            json={
                "id": "chatcmpl_123",
                "model": "gateway-model",
                "choices": [{"message": {"content": "final answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2},
            },
        )
    )
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

    assert route.called
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["cache-control"] == "no-cache, no-store, max-age=0"
    assert request.headers["pragma"] == "no-cache"
    assert json.loads(request.content) == {
        "model": "configured-model",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "temperature": 0.2,
        "max_tokens": 32,
    }
    assert response.text == "final answer"
    assert response.model == "gateway-model"
    assert response.request_id == "req_123"


@respx.mock
def test_cache_buster_uses_a_unique_url_parameter_without_changing_the_body() -> None:
    route = respx.post(
        url__regex=r"https://gateway\.example/chat/completions\?cache_bust=[0-9a-f]+$"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]},
        )
    )
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example",
            api_key="secret",
            model="model",
            cache_bust_query_param="cache_bust",
        )
    )

    client.complete("hello")

    assert route.called
    request = route.calls.last.request
    assert request.url.params["cache_bust"]
    assert json.loads(request.content)["messages"] == [{"role": "user", "content": "hello"}]


@respx.mock
def test_anthropic_completion_uses_messages_api() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_123",
                "model": "claude-test",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )
    )
    client = LLMClient(
        LLMConfig(
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_key="secret",
            model="claude-test",
            max_tokens=100,
        )
    )

    response = client.complete_messages(
        [
            {"role": "system", "content": "first system instruction"},
            {"role": "user", "content": "hello"},
        ],
        system="second system instruction",
    )

    assert route.called
    request = route.calls.last.request
    assert request.headers["x-api-key"] == "secret"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert json.loads(request.content) == {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
        "stream": False,
        "system": "second system instruction\n\nfirst system instruction",
    }
    assert response.text == "hello"
    assert response.finish_reason == "end_turn"


@respx.mock
def test_http_errors_include_status_and_body() -> None:
    respx.post("https://gateway.example/chat/completions").mock(
        return_value=httpx.Response(401, text='{"error":"bad key"}')
    )
    client = LLMClient(
        LLMConfig(
            provider="openai_compatible",
            base_url="https://gateway.example",
            api_key="bad",
            model="model",
        )
    )

    with pytest.raises(LLMRequestError) as exc_info:
        client.complete("hello")

    assert exc_info.value.status_code == 401
    assert exc_info.value.response_body == '{"error":"bad key"}'
