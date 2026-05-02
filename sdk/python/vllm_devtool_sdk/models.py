"""
Pydantic request/response models for the SDK.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    """Request body for chat completion."""
    model: str = ""
    messages: list[ChatMessage] = []
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = 512
    stream: bool = False
    stop: Optional[list[str]] = None


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage = ChatMessage()
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response from chat completion endpoint."""
    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = []
    usage: Usage = Usage()

    @property
    def text(self) -> str:
        """Get the text of the first choice."""
        if self.choices:
            return self.choices[0].message.content
        return ""


class ModelInfo(BaseModel):
    id: str = ""
    object: str = "model"
    created: int = 0
    owned_by: str = ""


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo] = []


class HealthResponse(BaseModel):
    status: str = ""
    model: Optional[str] = None
    reason: Optional[str] = None
