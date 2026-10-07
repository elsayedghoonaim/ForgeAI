"""
Typed local model manifest definition for ForgeAI.

Provides Pydantic models for GenerationDefaults, EngineSettings, KVCacheSettings,
and ForgeAIManifest, as specified in the ForgeAI Ollama/vLLM migration plan.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from forgeai.core.config import normalize_kv_cache_dtype, reject_gguf

_REPO_ID_ERROR = "Invalid HuggingFace repository ID format"
_REPO_ID_MAX_LENGTH = 200
_REPO_ID_MAX_PART_LENGTH = 96
_REPO_ID_PART_PATTERN = re.compile(r"^[a-zA-Z0-9]([a-zA-Z0-9._-]*[a-zA-Z0-9])?$")


def validate_repo_id_string(model_str: str) -> None:
    """Validate that a string is a safe Hugging Face repository ID."""
    if not model_str or not model_str.strip():
        raise ValueError(f"{_REPO_ID_ERROR}: repository ID cannot be empty.")
    if model_str != model_str.strip():
        raise ValueError(f"{_REPO_ID_ERROR}: surrounding whitespace is not allowed: {model_str!r}")

    model_lower = model_str.lower()
    reject_gguf(model_str)

    if model_lower.endswith(".bin") or model_lower.endswith(".pt"):
        raise ValueError(
            f"ERROR: PyTorch pickled weights (.bin/.pt) are unsupported (model: {model_str!r}). "
            "Remediation: Convert weights to safetensors or use a Hugging Face repository containing .safetensors weights."
        )

    if len(model_str) > _REPO_ID_MAX_LENGTH:
        raise ValueError(
            f"{_REPO_ID_ERROR}: exceeds {_REPO_ID_MAX_LENGTH} characters ({len(model_str)})."
        )

    if (
        "\\" in model_str
        or ".." in model_str
        or model_str.startswith("/")
        or model_str.startswith(":")
        or (len(model_str) > 1 and model_str[1] == ":")
    ):
        raise ValueError(f"{_REPO_ID_ERROR}: path-like ID rejected: {model_str!r}")

    parts = model_str.split("/")
    if len(parts) > 2 or any(not p for p in parts):
        raise ValueError(f"{_REPO_ID_ERROR}: expected 'name' or 'owner/name': {model_str!r}")

    for part in parts:
        if len(part) > _REPO_ID_MAX_PART_LENGTH:
            raise ValueError(
                f"{_REPO_ID_ERROR}: name part exceeds {_REPO_ID_MAX_PART_LENGTH} characters: "
                f"{model_str!r}"
            )
        if "--" in part:
            raise ValueError(f"{_REPO_ID_ERROR}: '--' is not allowed in names: {model_str!r}")
        if not _REPO_ID_PART_PATTERN.match(part):
            raise ValueError(f"{_REPO_ID_ERROR}: invalid characters: {model_str!r}")


_SAFETENSORS_CACHE: dict[tuple[str, int], bool] = {}
_SAFETENSORS_CACHE_MAX = 256


def _has_safetensors(dir_path: Path) -> bool:
    """Whether ``dir_path`` contains a ``*.safetensors`` file (positive results cached per mtime).

    The recursive scan is skipped while the directory's mtime is unchanged. Only positive
    results are cached so a directory that later gains weights is picked up immediately.
    """
    try:
        key = (str(dir_path.resolve()), dir_path.stat().st_mtime_ns)
    except OSError:
        return any(dir_path.rglob("*.safetensors"))
    if _SAFETENSORS_CACHE.get(key):
        return True
    found = any(dir_path.rglob("*.safetensors"))
    if found:
        if len(_SAFETENSORS_CACHE) >= _SAFETENSORS_CACHE_MAX:
            _SAFETENSORS_CACHE.clear()
        _SAFETENSORS_CACHE[key] = True
    return found


class GenerationDefaults(BaseModel):
    """Generation parameters for model inference."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=-1)
    max_tokens: int = Field(default=4096, ge=1)
    stop: list[str] = Field(default_factory=list)


class EngineSettings(BaseModel):
    """Engine runtime configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    tensor_parallel_size: int = Field(default=1, ge=1)
    pipeline_parallel_size: int = Field(default=1, ge=1)
    gpu_memory_utilization: float = Field(default=0.85, ge=0.1, le=0.98)
    enforce_eager: bool = Field(default=False)
    weight_quantization: Literal["none", "awq", "gptq", "fp8", "bitsandbytes"] = Field(
        default="none"
    )
    trust_remote_code: bool = Field(default=False)


class KVCacheSettings(BaseModel):
    """KV cache quantization and storage parameters."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    dtype: str = Field(default="auto")

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, value: Any) -> str:
        return normalize_kv_cache_dtype(value)


class ForgeAIManifest(BaseModel):
    """Canonical model manifest describing a local model tag or HuggingFace repo reference."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    schema_version: str = Field(default="2.0")
    source_kind: Literal["huggingface", "local_dir"] = Field(default="huggingface")
    name: str = Field(description="Tag name, e.g. gemma-2-9b-it:latest")
    model: str = Field(description="Hugging Face repo ID or local directory path")
    revision: str = Field(default="main")
    tokenizer_override: str | None = None
    system_prompt: str | None = None
    chat_template: str = Field(default="jinja")
    parameters: GenerationDefaults = Field(default_factory=GenerationDefaults)
    engine_settings: EngineSettings = Field(default_factory=EngineSettings)
    kv_cache: KVCacheSettings = Field(default_factory=KVCacheSettings)

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        model_str = self.model.strip()
        model_lower = model_str.lower()

        reject_gguf(model_str)

        if model_lower.endswith(".bin") or model_lower.endswith(".pt"):
            raise ValueError(
                f"ERROR: PyTorch pickled weights (.bin/.pt) are unsupported (model: {model_str!r}). "
                "Remediation: Convert weights to safetensors or use a Hugging Face repository containing .safetensors weights."
            )

        if self.source_kind == "huggingface":
            validate_repo_id_string(model_str)
        elif self.source_kind == "local_dir":
            dir_path = Path(model_str)
            if not dir_path.exists():
                raise ValueError(f"Local model directory does not exist: {model_str!r}")
            if not dir_path.is_dir():
                raise ValueError(f"Local model path is not a directory: {model_str!r}")

            config_file = dir_path / "config.json"
            if not config_file.exists():
                raise ValueError(
                    f"Local model directory missing required config.json: {model_str!r}"
                )

            if not _has_safetensors(dir_path):
                raise ValueError(
                    f"Local model directory missing .safetensors file: {model_str!r}"
                )

        return self

    def to_yaml(self) -> str:
        """Serialize manifest to deterministic YAML format."""
        data = self.model_dump(mode="json")
        return str(yaml.safe_dump(data, sort_keys=True, default_flow_style=False))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> Self:
        """Deserialize manifest from YAML string."""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a dictionary")
        res = cls.model_validate(data)
        return cast(Self, res)
