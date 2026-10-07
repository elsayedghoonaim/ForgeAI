"""OpenAI-style chat completion endpoints for the ForgeAI runtime."""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from forgeai.api.streaming import guarded_stream_response, sse_error_frames
from forgeai.core.backends.base import StreamStats, abort_request, iter_stream

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""

    @field_validator("content", mode="before")
    @classmethod
    def _normalize_content(cls, value: Any) -> Any:
        """Accept OpenAI content parts and null: text parts are concatenated, others ignored."""
        if value is None:
            return ""
        if isinstance(value, list):
            texts: list[str] = []
            for part in value:
                if isinstance(part, str):
                    texts.append(part)
                elif isinstance(part, dict) and part.get("type", "text") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        texts.append(text)
            return "".join(texts)
        return value


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=512, ge=1)
    stream: bool = False
    stop: list[str] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice] = []
    usage: Usage = Usage()


def _chat_events(
    request: Request,
    engine: Any,
    prompt: str,
    body: ChatCompletionRequest,
    request_id: str,
    model_name: str,
    stats: StreamStats,
) -> AsyncIterator[str]:
    """SSE event generator for a streaming chat completion (stats are per request)."""

    def _frame(delta: dict[str, Any], finish_reason: str | None = None) -> str:
        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def _events() -> AsyncIterator[str]:
        source = iter_stream(
            engine,
            prompt,
            stats,
            max_tokens=body.max_tokens or 512,
            temperature=body.temperature,
            top_p=body.top_p,
            stop=body.stop,
        )
        first = True
        async with contextlib.aclosing(source) as chunks:  # type: ignore[type-var]
            async for chunk in chunks:
                # OpenAI clients expect the role on the very first chunk
                yield _frame({"role": "assistant", "content": chunk} if first else {"content": chunk})
                first = False
        if first:  # empty completion: still announce the role
            yield _frame({"role": "assistant", "content": ""})
        request.state.prompt_tokens = stats.prompt_tokens
        request.state.completion_tokens = stats.completion_tokens
        yield _frame({}, stats.finish_reason or "stop")
        yield "data: [DONE]\n\n"

    return _events()


async def _sse_on_error(err: Exception) -> list[str]:
    return sse_error_frames(err)


@router.post("/chat/completions", response_model=None)
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """OpenAI-compatible chat completion endpoint."""

    runtime_adapter = getattr(request.app.state, "runtime_adapter", None)
    if runtime_adapter is not None:
        model_tag = body.model or "latest"
        try:
            lease, manifest, record, key = await runtime_adapter.acquire_lease(model_tag)
        except KeyError as err:
            raise HTTPException(
                status_code=404, detail=f"Model '{body.model}' not found"
            ) from err
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        acquired = True
        try:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            messages = [message.model_dump() for message in body.messages]

            if body.stream:
                if hasattr(lease.engine, "supports_streaming") and not lease.engine.supports_streaming:
                    raise HTTPException(status_code=501, detail="Streaming not supported by this backend")
                prompt = lease.engine.build_prompt(messages)

                stats = StreamStats(request_id=f"stream-{uuid.uuid4().hex}")
                events = _chat_events(
                    request,
                    lease.engine,
                    prompt,
                    body,
                    request_id,
                    body.model or record.ref.full_tag,
                    stats,
                )
                response = guarded_stream_response(
                    events,
                    release=lambda: runtime_adapter.release_lease(key),
                    abort=lambda: abort_request(lease.engine, stats.request_id),
                    on_error=_sse_on_error,
                    media_type="text/event-stream",
                )
                acquired = False  # Ownership transferred to the guarded response
                return response

            prompt = lease.engine.build_prompt(messages)
            result = await lease.engine.generate(
                prompt=prompt,
                max_tokens=body.max_tokens or 512,
                temperature=body.temperature,
                top_p=body.top_p,
                stop=body.stop,
            )
            request.state.prompt_tokens = getattr(result, "prompt_tokens", 0)
            request.state.completion_tokens = getattr(result, "completion_tokens", 0)

            return ChatCompletionResponse(
                id=request_id,
                created=int(time.time()),
                model=body.model or record.ref.full_tag,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessage(role="assistant", content=getattr(result, "text", "")),
                        finish_reason=getattr(result, "finish_reason", "stop") or "stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=getattr(result, "prompt_tokens", 0),
                    completion_tokens=getattr(result, "completion_tokens", 0),
                    total_tokens=getattr(result, "total_tokens", 0),
                ),
            )
        finally:
            if acquired:
                await runtime_adapter.release_lease(key)

    # Legacy engine fallback path
    engine = request.app.state.engine
    if engine is None or not engine.is_running:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if body.stream:
        if not engine.supports_streaming:
            raise HTTPException(status_code=501, detail="Streaming not supported by this backend")

        async def _noop_release() -> None:
            return None

        legacy_stats = StreamStats(request_id=f"stream-{uuid.uuid4().hex}")
        legacy_prompt = engine.build_prompt([message.model_dump() for message in body.messages])
        return guarded_stream_response(
            _chat_events(
                request,
                engine,
                legacy_prompt,
                body,
                f"chatcmpl-{uuid.uuid4().hex[:12]}",
                body.model or engine.settings.model_name,
                legacy_stats,
            ),
            release=_noop_release,
            abort=lambda: abort_request(engine, legacy_stats.request_id),
            on_error=_sse_on_error,
            media_type="text/event-stream",
        )

    prompt = engine.build_prompt([message.model_dump() for message in body.messages])
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    result = await engine.generate(
        prompt=prompt,
        max_tokens=body.max_tokens or 512,
        temperature=body.temperature,
        top_p=body.top_p,
        stop=body.stop,
    )
    request.state.prompt_tokens = result.prompt_tokens
    request.state.completion_tokens = result.completion_tokens

    return ChatCompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=body.model or engine.settings.model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=result.text),
                finish_reason=result.finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        ),
    )
