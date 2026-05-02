# vLLM DevTool Python SDK

This SDK provides an async Python client for the vLLM DevTool HTTP API.

It is intended for:

- application integration
- health and readiness checks
- model listing
- non-streaming chat completions against the built-in API

## Installation

```bash
pip install forgeai-sdk
```

## What The SDK Talks To

The SDK is designed for the API started by:

```bash
forgeai serve MODEL
```

Default base URL:

```text
http://localhost:8000
```

## Quick Start

```python
import asyncio

from forgeai_sdk import DevToolClient


async def main() -> None:
    async with DevToolClient("http://localhost:8000") as client:
        health = await client.health()
        print(health.status)

        models = await client.list_models()
        print([model.id for model in models.data])

        response = await client.chat(
            "Explain quantum computing simply.",
            model="google/gemma-4-E2B-it",
        )
        print(response.choices[0].message.content)


asyncio.run(main())
```

## Authentication

Use an API key:

```python
client = DevToolClient(
    "http://localhost:8000",
    api_key="vdt_your_api_key_here",
)
```

Or a bearer token:

```python
client = DevToolClient(
    "http://localhost:8000",
    token="your_jwt_here",
)
```

## API Overview

`DevToolClient` currently provides:

- `chat(...)`
- `chat_stream(...)`
- `list_models()`
- `health()`
- `ready()`
- `metrics()`
- `close()`

## `DevToolClient`

Constructor:

```python
DevToolClient(
    base_url: str = "http://localhost:8000",
    api_key: str | None = None,
    token: str | None = None,
    timeout: float = 120.0,
)
```

Arguments:

- `base_url`: API root URL
- `api_key`: sends `X-API-Key`
- `token`: sends `Authorization: Bearer ...`
- `timeout`: HTTP timeout in seconds

The client is async-context-manager friendly:

```python
async with DevToolClient("http://localhost:8000") as client:
    ...
```

## Chat Completions

### `chat(...)`

```python
response = await client.chat(
    message="Summarize transformers in one paragraph.",
    model="google/gemma-4-E2B-it",
    system="You are concise.",
    temperature=0.7,
    max_tokens=512,
)
```

Parameters:

- `message`: user message
- `model`: optional model override; leave empty if the server has a default model
- `system`: optional system prompt
- `temperature`: sampling temperature
- `max_tokens`: generation cap
- `history`: optional list of previous `ChatMessage` objects

The SDK assembles the OpenAI-style `messages` payload for you.

### `chat_stream(...)`

The client includes a streaming helper:

```python
async for token in client.chat_stream("Write a short poem about latency."):
    print(token, end="", flush=True)
```

Important current limitation:

- the built-in `forgeai serve` API does not yet support HTTP streaming chat completions
- the current server returns `501` if `stream=true`
- use `chat()` for the built-in API today

Keep `chat_stream()` only if you are targeting a compatible future server or a custom streaming-compatible HTTP layer.

## Models

### `list_models()`

```python
models = await client.list_models()
for model in models.data:
    print(model.id)
```

This maps to:

- `GET /v1/models`

## Health and Readiness

### `health()`

Maps to:

- `GET /healthz`

### `ready()`

Maps to:

- `GET /readyz`

Example:

```python
health = await client.health()
ready = await client.ready()
print(health.status, ready.status)
```

## Metrics

### `metrics()`

Returns the Prometheus metrics text payload:

```python
metrics_text = await client.metrics()
print(metrics_text[:500])
```

Maps to:

- `GET /metrics`

If server auth is enabled, your client must provide credentials with access to that endpoint.

## Error Handling

The client uses `httpx` and calls `raise_for_status()` for most API operations, so standard `httpx.HTTPStatusError` handling applies.

Example:

```python
import httpx

try:
    await client.list_models()
except httpx.HTTPStatusError as exc:
    print(exc.response.status_code, exc.response.text)
```

## Related Documentation

- [../../README.md](../../README.md)
- [../../docs/WSL.md](../../docs/WSL.md)
