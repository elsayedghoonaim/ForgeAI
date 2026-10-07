from __future__ import annotations

from forgeai.core.backends.base import BaseBackend
from forgeai.core.config import DevToolSettings, reject_gguf


def create_backend(settings: DevToolSettings, streaming: bool = False, quiet_startup: bool = False) -> BaseBackend:
    """Create and return the vLLM backend (the only supported backend)."""
    reject_gguf(settings.model_path or settings.model_name)

    from forgeai.core.backends.vllm_backend import VLLMBackend
    return VLLMBackend(settings, streaming=streaming, quiet_startup=quiet_startup)
