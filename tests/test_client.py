"""Unit tests for vllm_client/client.py.

Uses ``respx`` to intercept httpx requests so no real network call is made.
All tests are async; pytest-asyncio picks them up automatically via
``asyncio_mode = "auto"`` set in pyproject.toml.
"""

import json

import httpx
import pytest
import respx

from vllm_client.client import VllmOmniClient

BASE_URL = "http://localhost:8000/v1"

_CHAT_RESPONSE = {
    "choices": [{"message": {"content": "Hello, world!"}}]
}
_MODELS_RESPONSE = {
    "data": [{"id": "test-model"}, {"id": "other-model"}]
}


# ---------------------------------------------------------------------------
# chat_completion
# ---------------------------------------------------------------------------

@respx.mock
async def test_chat_completion_returns_parsed_response():
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    async with VllmOmniClient(BASE_URL) as client:
        result = await client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
        )
    assert result == _CHAT_RESPONSE


@respx.mock
async def test_chat_completion_sends_correct_payload():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    async with VllmOmniClient(BASE_URL) as client:
        await client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="my-model",
            temperature=0.7,
        )
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "my-model"
    assert payload["temperature"] == 0.7
    assert payload["messages"][0]["content"] == "Hi"


@respx.mock
async def test_chat_completion_forwards_extra_kwargs():
    route = respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT_RESPONSE)
    )
    async with VllmOmniClient(BASE_URL) as client:
        await client.chat_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="test-model",
            max_tokens=256,
            top_p=0.9,
        )
    payload = json.loads(route.calls.last.request.content)
    assert payload["max_tokens"] == 256
    assert payload["top_p"] == 0.9


@respx.mock
async def test_chat_completion_raises_on_server_error():
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    async with VllmOmniClient(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
            )


@respx.mock
async def test_chat_completion_raises_on_auth_error():
    respx.post(f"{BASE_URL}/chat/completions").mock(
        return_value=httpx.Response(401, text="Unauthorized")
    )
    async with VllmOmniClient(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="test-model",
            )


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

@respx.mock
async def test_list_models_returns_data_list():
    respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=_MODELS_RESPONSE)
    )
    async with VllmOmniClient(BASE_URL) as client:
        models = await client.list_models()
    assert models == _MODELS_RESPONSE["data"]


@respx.mock
async def test_list_models_returns_empty_list_when_data_missing():
    respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json={})
    )
    async with VllmOmniClient(BASE_URL) as client:
        models = await client.list_models()
    assert models == []


@respx.mock
async def test_list_models_raises_on_http_error():
    respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(403, text="Forbidden")
    )
    async with VllmOmniClient(BASE_URL) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.list_models()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@respx.mock
async def test_context_manager_closes_client():
    respx.get(f"{BASE_URL}/models").mock(
        return_value=httpx.Response(200, json=_MODELS_RESPONSE)
    )
    client = VllmOmniClient(BASE_URL)
    async with client as c:
        assert c is client
        await c.list_models()
    # After __aexit__ the underlying httpx client should be closed
    assert client._http.is_closed
