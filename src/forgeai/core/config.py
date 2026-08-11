"""Configuration management for the ForgeAI vLLM runtime."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class BackendType(str, Enum):
    """Known backend identifiers."""

    VLLM = "vllm"


class QuantizationType(str, Enum):
    """Known quantization identifiers."""

    NONE = "none"
    AWQ = "awq"
    GPTQ = "gptq"
    FP8 = "fp8"
    BITSANDBYTES = "bitsandbytes"
    AUTO = "auto"


class KVCacheSettings(BaseSettings):
    """vLLM KV-cache quantization configuration settings."""

    dtype: str = Field(default="auto", description="KV cache quantization format")

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        val = value.lower().strip() if isinstance(value, str) else value
        allowed = {"auto", "fp8", "turboquant_k8v4", "turboquant_4bit_nc"}
        if val == "turboquant_3bit_nc":
            raise ValueError(
                "turboquant_3bit_nc is an aggressive POC-only profile and is not accepted in normal runtime config."
            )
        if val not in allowed:
            raise ValueError(
                f"Invalid kv_cache_dtype '{value}'. Must be one of {sorted(allowed)}."
            )
        return val


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
    tokenizer: str | None = Field(default=None, description="Custom tokenizer model or directory path")
    revision: str | None = Field(default=None, description="Hugging Face model revision or commit hash")
    max_model_len: int | None = Field(default=None, ge=1, description="Maximum context length")
    trust_remote_code: bool = Field(default=False, description="Allow remote code execution")



    # --- Backend ---
    backend: BackendType | None = Field(default=BackendType.VLLM, description="Inference backend")
    quantization: QuantizationType = Field(
        default=QuantizationType.AUTO,
        description="Quantization format",
    )

    # --- GPU / Parallelism ---
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel GPUs")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallel GPUs")
    dtype: str = Field(default="auto", description="Model weights data type")
    gpu_memory_utilization: float = Field(
        default=0.85,
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
    max_num_seqs: int = Field(default=4, ge=1, description="Max concurrent sequences")

    # --- Shared Engine Lifecycle & Admission ---
    max_loaded_models: int = Field(default=1, ge=1, description="Max loaded engines in VRAM")
    load_concurrency: int = Field(default=1, ge=1, description="Model load concurrency limit")
    request_queue_depth: int = Field(default=32, ge=1, description="Max queued load requests")
    default_keep_alive: str = Field(default="5m", description="Default engine keep_alive TTL")
    kv_cache_dtype: str = Field(default="auto", description="KV cache quantization format")

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

    @field_validator("kv_cache_dtype", mode="before")
    @classmethod
    def validate_kv_cache_dtype(cls, value: str) -> str:
        val = value.lower().strip() if isinstance(value, str) else value
        allowed = {"auto", "fp8", "turboquant_k8v4", "turboquant_4bit_nc"}
        if val == "turboquant_3bit_nc":
            raise ValueError(
                "turboquant_3bit_nc is an aggressive POC-only profile and is not accepted in normal runtime config."
            )
        if val not in allowed:
            raise ValueError(
                f"Invalid kv_cache_dtype '{value}'. Must be one of {sorted(allowed)}."
            )
        return val

    @model_validator(mode="before")
    @classmethod
    def reject_legacy_llamacpp_options(cls, values: Any) -> Any:
        """Reject removed llama.cpp options and legacy backend choices with an actionable error."""
        if not isinstance(values, dict):
            return values

        legacy_fields = {"n_gpu_layers", "n_ctx", "n_batch", "chat_format"}
        lowered_keys = {str(k).lower(): k for k in values.keys()}
        detected_legacy = [lowered_keys[k] for k in legacy_fields if k in lowered_keys]

        backend_val = values.get("backend")
        backend_str = ""
        if isinstance(backend_val, str):
            backend_str = backend_val.lower().strip()
        elif hasattr(backend_val, "value"):
            backend_str = str(backend_val.value).lower().strip()

        is_legacy_backend = bool(backend_str) and backend_str != "vllm"

        quant_val = values.get("quantization")
        quant_str = ""
        if isinstance(quant_val, str):
            quant_str = quant_val.lower().strip()
        elif hasattr(quant_val, "value"):
            quant_str = str(quant_val.value).lower().strip()

        is_legacy_quant = quant_str == "gguf"

        if is_legacy_quant:
            raise ValueError(
                f"ERROR: GGUF model format and quantization variants are unsupported in ForgeAI v2.0+ (quantization={quant_val!r}). "
                "llama.cpp has been removed in favor of vLLM. "
                "Remediation: Specify Hugging Face repo IDs or local safetensors directories and vLLM-compatible settings."
            )

        if detected_legacy or is_legacy_backend:
            reasons = []
            if is_legacy_backend:
                reasons.append(f"backend={backend_val!r}")
            if detected_legacy:
                reasons.append(f"legacy fields: {sorted(detected_legacy)}")

            raise ValueError(
                f"ERROR: llama.cpp backend and legacy options have been removed in ForgeAI v2.0+ ({', '.join(reasons)}). "
                "vLLM is now the sole inference engine. "
                "Remediation: Specify Hugging Face repo IDs or local safetensors directories and vLLM-compatible settings."
            )
        return values

    @model_validator(mode="after")
    def validate_runtime_scope(self) -> DevToolSettings:
        """Validate backend consistency and concurrency constraints."""
        model_str = (self.model_path or self.model_name).strip()
        if model_str and (".gguf" in model_str.lower() or model_str.lower().endswith(".gguf")):
            raise ValueError(
                f"ERROR: GGUF model format is unsupported in ForgeAI v2.0+ (model: {model_str!r}). "
                "llama.cpp has been removed in favor of vLLM. "
                "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
            )
        if self.load_concurrency > self.max_loaded_models:
            raise ValueError(
                f"load_concurrency ({self.load_concurrency}) cannot exceed max_loaded_models ({self.max_loaded_models})."
            )
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
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "trust_remote_code": self.trust_remote_code,
            "kv_cache_dtype": self.kv_cache_dtype,
        }
        if self.tokenizer:
            kwargs["tokenizer"] = self.tokenizer
        if self.revision:
            kwargs["revision"] = self.revision
        if self.enforce_eager:
            kwargs["enforce_eager"] = True

        if self.max_num_batched_tokens:
            kwargs["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.max_model_len:
            kwargs["max_model_len"] = self.max_model_len
        if self.quantization not in (
            QuantizationType.AUTO,
            QuantizationType.NONE,
        ):
            kwargs["quantization"] = self.quantization.value
        return kwargs
