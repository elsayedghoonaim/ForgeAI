"""Configuration management for the ForgeAI dual-backend runtime."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class BackendType(str, Enum):
    """Known backend identifiers kept for compatibility with older configs."""

    VLLM = "vllm"
    LLAMA_CPP = "llama_cpp"
    AUTO = "auto"


class QuantizationType(str, Enum):
    """Known quantization identifiers kept for compatibility with older configs."""

    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    GGUF = "gguf"
    AUTO = "auto"


class DevToolSettings(BaseSettings):
    """
    Central configuration for vLLM DevTool.

    All settings can be overridden via environment variables prefixed with
    ``forgeai_``.
    """

    model_config = {"env_prefix": "forgeai_", "case_sensitive": False}

    # --- Model ---
    model_name: str = Field(default="", description="Model name or HuggingFace repo ID")
    model_path: str | None = Field(default=None, description="Local path to model weights")
    max_model_len: int | None = Field(default=None, ge=1, description="Maximum context length")
    trust_remote_code: bool = Field(default=False, description="Allow remote code execution")

    # --- Backend ---
    backend: BackendType | None = Field(default=BackendType.AUTO, description="Inference backend")
    quantization: QuantizationType = Field(
        default=QuantizationType.AUTO,
        description="Quantization format",
    )

    # --- GPU / Parallelism ---
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel GPUs")
    gpu_memory_utilization: float = Field(
        default=0.90,
        ge=0.1,
        le=1.0,
        description="Target GPU memory utilization",
    )
    enforce_eager: bool = Field(
        default=False,
        description="Favor eager execution over compile-heavy startup",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum tokens scheduled in a single batch",
    )
    max_num_seqs: int = Field(default=256, ge=1, description="Max concurrent sequences")

    # --- llama.cpp Specific ---
    n_gpu_layers: int = Field(default=0, description="GPU layers to offload (-1 = all)")
    n_ctx: int = Field(default=4096, ge=128, description="Context window size")
    n_batch: int = Field(default=512, ge=1, description="Batch size for prompt processing")
    chat_format: str | None = Field(default=None, description="Chat template override")

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, ge=1, le=65535, description="API server port")
    request_id_header: str = Field(default="X-Request-ID", description="Request ID header name")
    log_json: bool = Field(default=False, description="Emit JSON logs instead of Rich logs")

    # --- Paths ---
    cache_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
        description="Model cache directory",
    )
    profiles_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"),
            ".forgeai",
            "profiles",
        ),
        description="Deployment profiles directory",
    )
    audit_log_dir: str = Field(
        default_factory=lambda: os.path.join(
            os.path.expanduser("~"),
            ".forgeai",
            "audit",
        ),
        description="Audit log directory",
    )

    # --- Telemetry ---
    telemetry_enabled: bool = Field(default=False, description="Enable opt-in telemetry")

    # --- Security ---
    enforce_version_check: bool = Field(default=True, description="Block vulnerable vLLM versions")
    enable_safety_scan: bool = Field(default=True, description="Run a post-download safety scan")
    audit_logging_enabled: bool = Field(default=True, description="Enable audit logging")
    rate_limit_enabled: bool = Field(default=True, description="Enable in-memory rate limiting")
    rate_limit_requests: int = Field(
        default=120,
        ge=1,
        description="Allowed requests per rate-limit window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        description="Rate-limit window length in seconds",
    )

    # --- Auth ---
    auth_enabled: bool = Field(default=False, description="Enable API authentication")
    auth_secret_key: str = Field(
        default="change-me-in-production",
        description="JWT signing secret",
    )
    auth_algorithm: str = Field(default="HS256", description="JWT algorithm")
    auth_token_expire_minutes: int = Field(default=60, description="Token expiration in minutes")
    bootstrap_api_key: str | None = Field(
        default=None,
        description="Bootstrap API key to register on startup",
    )
    bootstrap_api_key_name: str = Field(default="bootstrap", description="Bootstrap API key label")
    bootstrap_api_key_role: str = Field(default="admin", description="Bootstrap API key role")

    @field_validator("model_name", mode="before")
    @classmethod
    def strip_model_name(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("request_id_header", mode="before")
    @classmethod
    def normalize_request_id_header(cls, value: str) -> str:
        header = value.strip() if isinstance(value, str) else value
        return header or "X-Request-ID"

    @model_validator(mode="after")
    def validate_runtime_scope(self) -> DevToolSettings:
        """Validate backend consistency."""
        return self

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""

        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.profiles_dir).mkdir(parents=True, exist_ok=True)
        Path(self.audit_log_dir).mkdir(parents=True, exist_ok=True)

    def to_vllm_kwargs(self) -> dict[str, object]:
        """Convert settings to vLLM engine keyword arguments."""

        kwargs: dict[str, object] = {
            "model": self.model_path or self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.enforce_eager:
            kwargs["enforce_eager"] = True
        if self.max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        if self.quantization not in (
            QuantizationType.AUTO,
            QuantizationType.NONE,
            QuantizationType.GGUF,
        ):
            kwargs["quantization"] = self.quantization.value
        return kwargs

    def to_llamacpp_kwargs(self) -> dict[str, object]:
        """Convert settings to llama-cpp-python engine keyword arguments."""
        return {
            "model_path": self.model_path or self.model_name,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "chat_format": self.chat_format,
            "verbose": False,
        }
