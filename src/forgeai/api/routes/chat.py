"""OpenAI-style chat completion endpoints for the ForgeAI runtime."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter()


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


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
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Model '{body.model}' not found")

        acquired = True
        try:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            messages = [message.model_dump() for message in body.messages]

            if body.stream:
                if hasattr(lease.engine, "supports_streaming") and not lease.engine.supports_streaming:
                    raise HTTPException(status_code=501, detail="Streaming not supported by this backend")
                prompt = lease.engine.build_prompt(messages)

                acquired = False  # Ownership transferred to generator finally
                async def _runtime_stream_generator():
                    try:
                        async for chunk in lease.engine.generate_stream(
                            prompt=prompt,
                            max_tokens=body.max_tokens or 512,
                            temperature=body.temperature,
                            top_p=body.top_p,
                            stop=body.stop,
                        ):
                            payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": body.model or record.ref.full_tag,
                                "choices": [{"index": 0, "delta": {"content": chunk}}],
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        await runtime_adapter.release_lease(key)

                return StreamingResponse(_runtime_stream_generator(), media_type="text/event-stream")


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

        async def _stream_generator():
            prompt = engine.build_prompt([message.model_dump() for message in body.messages])
            request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            async for chunk in engine.generate_stream(
                prompt=prompt,
                max_tokens=body.max_tokens or 512,
                temperature=body.temperature,
                top_p=body.top_p,
                stop=body.stop,
            ):

                payload = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.model or engine.settings.model_name,
                    "choices": [{"index": 0, "delta": {"content": chunk}}]
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_generator(), media_type="text/event-stream")

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
