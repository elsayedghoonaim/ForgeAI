"""Ollama-compatible API request and response schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OllamaOptions(BaseModel):
    """Ollama-style generation options."""

    model_config = ConfigDict(extra="ignore")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=-1)
    num_predict: int | None = Field(default=None, ge=1)
    stop: list[str] | str | None = None


class OllamaGenerateRequest(BaseModel):
    """Request schema for POST /api/generate."""

    model_config = ConfigDict(extra="ignore")

    model: str
    prompt: str = ""
    suffix: str | None = None
    system: str | None = None
    template: str | None = None
    stream: bool = True
    raw: bool = False
    keep_alive: Any = None
    options: dict[str, Any] | OllamaOptions | None = None
    context: list[int] | None = None
    images: list[str] | None = None

    @model_validator(mode="after")
    def validate_request(self) -> OllamaGenerateRequest:
        if not self.model or not self.model.strip():
            raise ValueError("Model tag cannot be empty.")
        if isinstance(self.options, dict):
            OllamaOptions.model_validate(self.options)
        return self


class OllamaChatMessage(BaseModel):
    """Message item for POST /api/chat."""

    model_config = ConfigDict(extra="ignore")

    role: str = "user"
    content: str = ""
    images: list[str] | None = None


class OllamaChatRequest(BaseModel):
    """Request schema for POST /api/chat."""

    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[OllamaChatMessage]
    stream: bool = True
    keep_alive: Any = None
    options: dict[str, Any] | OllamaOptions | None = None

    @model_validator(mode="after")
    def validate_request(self) -> OllamaChatRequest:
        if not self.model or not self.model.strip():
            raise ValueError("Model tag cannot be empty.")
        if isinstance(self.options, dict):
            OllamaOptions.model_validate(self.options)
        return self


class OllamaEmbedRequest(BaseModel):
    """Request schema for POST /api/embed."""

    model_config = ConfigDict(extra="ignore")

    model: str
    input: str | list[str]
    keep_alive: Any = None
    options: dict[str, Any] | None = None
    truncate: bool | None = None
    dimensions: int | None = None

    @model_validator(mode="after")
    def validate_request(self) -> OllamaEmbedRequest:
        if not self.model or not self.model.strip():
            raise ValueError("Model tag cannot be empty.")
        return self



class OllamaShowRequest(BaseModel):
    """Request schema for POST /api/show."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    model: str | None = None

    @property
    def model_name(self) -> str:
        res = self.model or self.name or ""
        if not res.strip():
            raise ValueError("Model tag is required in 'model' or 'name' field.")
        return res.strip()


class OllamaPullRequest(BaseModel):
    """Request schema for POST /api/pull."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    model: str | None = None
    insecure: bool = False
    username: str | None = None
    password: str | None = None
    stream: bool = True

    @property
    def model_name(self) -> str:
        res = self.model or self.name or ""
        if not res.strip():
            raise ValueError("Model tag is required in 'model' or 'name' field.")
        return res.strip()


class OllamaDeleteRequest(BaseModel):
    """Request schema for POST /api/delete."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    model: str | None = None

    @property
    def model_name(self) -> str:
        res = self.model or self.name or ""
        if not res.strip():
            raise ValueError("Model tag is required in 'model' or 'name' field.")
        return res.strip()
