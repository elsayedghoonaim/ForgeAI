from __future__ import annotations

from forgeai.core.backends.base import BaseBackend
from forgeai.core.config import BackendType, DevToolSettings


def resolve_backend(settings: DevToolSettings) -> BackendType:
    """Resolve the appropriate backend based on settings."""
    model = settings.model_path or settings.model_name
    if model and (".gguf" in model.lower() or model.lower().endswith(".gguf")):
        raise ValueError(
            f"ERROR: GGUF model format is unsupported in ForgeAI v2.0+ (model: {model!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )
    return BackendType.VLLM


def create_backend(settings: DevToolSettings, streaming: bool = False, quiet_startup: bool = False) -> BaseBackend:
    """Create and return the vLLM backend."""
    model = settings.model_path or settings.model_name
    if model and (".gguf" in model.lower() or model.lower().endswith(".gguf")):
        raise ValueError(
            f"ERROR: GGUF model format is unsupported in ForgeAI v2.0+ (model: {model!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )

    from forgeai.core.backends.vllm_backend import VLLMBackend
    return VLLMBackend(settings, streaming=streaming, quiet_startup=quiet_startup)
