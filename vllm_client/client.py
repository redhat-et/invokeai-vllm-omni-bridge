"""Async HTTP client for the vLLM-Omni OpenAI-compatible API.

All methods are ``async`` and must be awaited.  The client is designed to be
used as an async context manager so the underlying ``httpx.AsyncClient`` is
properly closed after use::

    async with VllmOmniClient(base_url="http://localhost:8000/v1") as client:
        response = await client.chat_completion(messages=[...], model="Qwen/Qwen2-Audio-7B-Instruct")

Alternatively, call ``await client.aclose()`` explicitly when done.
"""

import httpx

from typing import Any


class VllmOmniClient:
    """Thin async wrapper around vLLM-Omni's OpenAI-compatible REST API.

    Args:
        base_url: Base URL of the vLLM server including the ``/v1`` prefix,
            e.g. ``"http://localhost:8000/v1"``.
        api_key: Bearer token sent in the ``Authorization`` header.  vLLM
            servers that require no auth accept ``"EMPTY"`` by convention.
        timeout: Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "VllmOmniClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Send a chat completion request to ``POST /chat/completions``.

        Args:
            messages: List of OpenAI-format message dicts, e.g.
                ``[{"role": "user", "content": "Hello"}]``.  Content may be a
                string or a list of content parts (text + image_url) for
                multimodal inputs.
            model: Model name as served by vLLM, e.g.
                ``"Qwen/Qwen2-Audio-7B-Instruct"``.
            **kwargs: Extra fields forwarded verbatim to the request body
                (``temperature``, ``max_tokens``, ``stream``, etc.).

        Returns:
            Parsed JSON response body as a dict (OpenAI ``ChatCompletion``
            schema).

        Raises:
            httpx.HTTPStatusError: If the server returns a 4xx or 5xx status.
            httpx.TimeoutException: If the request exceeds the configured timeout.
        """
        payload: dict[str, Any] = {"model": model, "messages": messages, **kwargs}
        response = await self._http.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch the list of models available on the vLLM server (``GET /models``).

        Useful for connectivity checks and for discovering the correct model
        name to pass to ``chat_completion()``.

        Returns:
            List of model objects from the OpenAI ``Model`` schema.

        Raises:
            httpx.HTTPStatusError: If the server returns a 4xx or 5xx status.
            httpx.TimeoutException: If the request exceeds the configured timeout.
        """
        response = await self._http.get("/models")
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data.get("data", [])
