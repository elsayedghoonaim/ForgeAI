"""OpenAI-style chat completion endpoints for the ForgeAI runtime."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
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


@router.post("/chat/completions")
async def create_chat_completion(
    request: Request,
    body: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint."""

    engine = request.app.state.engine
    if engine is None or not engine.is_running:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if body.stream:
        if not engine.supports_streaming:
            raise HTTPException(status_code=501, detail="Streaming not supported by this backend")
        from fastapi.responses import StreamingResponse

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
    result = engine.generate(
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
