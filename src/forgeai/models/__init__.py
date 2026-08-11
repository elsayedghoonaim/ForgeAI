"""
Model Intelligence — discovery, loading, quantization, adapters, safety, manifests, and registry.
"""

from forgeai.models.loader import CacheManager, download_model, get_cached_models
from forgeai.models.manifest import (
    EngineSettings,
    ForgeAIManifest,
    GenerationDefaults,
    KVCacheSettings,
)
from forgeai.models.registry import (
    ModelInfo,
    ModelRecord,
    ModelRef,
    ModelRegistry,
    discover_model,
    is_multimodal,
)
from forgeai.models.zoo import MODEL_ALIASES, resolve_model_name

__all__ = [
    "MODEL_ALIASES",
    "resolve_model_name",
    "discover_model",
    "is_multimodal",
    "ModelInfo",
    "ModelRef",
    "ModelRecord",
    "ModelRegistry",
    "CacheManager",
    "download_model",
    "get_cached_models",
    "ForgeAIManifest",
    "GenerationDefaults",
    "EngineSettings",
    "KVCacheSettings",
]
