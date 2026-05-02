from __future__ import annotations

from forgeai.core.backends.base import BaseBackend
from forgeai.core.config import BackendType, DevToolSettings


def resolve_backend(settings: DevToolSettings) -> BackendType:
    """Resolve the appropriate backend based on settings and model format."""

    # 1. Explicit setting
    if settings.backend is not None and settings.backend != BackendType.AUTO:
        return settings.backend

    # 2. Auto-detect from model
    model = settings.model_path or settings.model_name
    if model and model.endswith(".gguf"):
        return BackendType.LLAMA_CPP

    # Default fallback
    return BackendType.VLLM


def create_backend(settings: DevToolSettings, streaming: bool = False, quiet_startup: bool = False) -> BaseBackend:
    """Create and return the resolved backend."""

    backend_type = resolve_backend(settings)

    if backend_type == BackendType.VLLM:
        from forgeai.core.backends.vllm_backend import VLLMBackend
        return VLLMBackend(settings, streaming=streaming, quiet_startup=quiet_startup)
    elif backend_type == BackendType.LLAMA_CPP:
        from forgeai.core.backends.llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend(settings, streaming=streaming, quiet_startup=quiet_startup)

    raise ValueError(f"Unknown backend type: {backend_type}")
