"""
DevToolClient — Async HTTP client for the vLLM DevTool API.

Provides a high-level interface for chat completions, model management,
and health checks with automatic retry and streaming support.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

import httpx

from forgeai_sdk.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    HealthResponse,
    ModelListResponse,
)


class DevToolClient:
    """
    Async client for the vLLM DevTool API.

    Example:
        ```python
        async with DevToolClient("http://localhost:8000") as client:
            response = await client.chat("Hello, how are you?")
            print(response.text)
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        elif token:
            headers["Authorization"] = f"Bearer {token}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def __aenter__(self) -> "DevToolClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    # --- Chat ---

    async def chat(
        self,
        message: str,
        model: str = "",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        history: Optional[list[ChatMessage]] = None,
    ) -> ChatCompletionResponse:
        """
        Send a chat completion request.

        Args:
            message: User message.
            model: Model name (optional if server has a default).
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            history: Previous conversation messages.
        """
        messages = []
        if system:
            messages.append(ChatMessage(role="system", content=system))
        if history:
            messages.extend(history)
        messages.append(ChatMessage(role="user", content=message))

        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        resp = await self._client.post(
            "/v1/chat/completions",
            json=request.model_dump(),
        )
        resp.raise_for_status()
        return ChatCompletionResponse(**resp.json())

    async def chat_stream(
        self,
        message: str,
        model: str = "",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    # --- Models ---

    async def list_models(self) -> ModelListResponse:
        """List available models."""
        resp = await self._client.get("/v1/models")
        resp.raise_for_status()
        return ModelListResponse(**resp.json())

    # --- Health ---

    async def health(self) -> HealthResponse:
        """Check server liveness."""
        resp = await self._client.get("/healthz")
        return HealthResponse(**resp.json())

    async def ready(self) -> HealthResponse:
        """Check server readiness."""
        resp = await self._client.get("/readyz")
        return HealthResponse(**resp.json())

    # --- Metrics ---

    async def metrics(self) -> str:
        """Get Prometheus metrics."""
        resp = await self._client.get("/metrics")
        return resp.text
