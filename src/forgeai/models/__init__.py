"""Model Intelligence — discovery, loading, quantization, adapters, safety, manifests, and registry.

Imports are lazy (PEP 562) so importing a submodule does not load the whole package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forgeai.models.loader import (
        CacheManager as CacheManager,
    )
    from forgeai.models.loader import (
        download_model as download_model,
    )
    from forgeai.models.loader import (
        get_cached_models as get_cached_models,
    )
    from forgeai.models.manifest import (
        EngineSettings as EngineSettings,
    )
    from forgeai.models.manifest import (
        ForgeAIManifest as ForgeAIManifest,
    )
    from forgeai.models.manifest import (
        GenerationDefaults as GenerationDefaults,
    )
    from forgeai.models.manifest import (
        KVCacheSettings as KVCacheSettings,
    )
    from forgeai.models.registry import (
        ModelInfo as ModelInfo,
    )
    from forgeai.models.registry import (
        ModelRecord as ModelRecord,
    )
    from forgeai.models.registry import (
        ModelRef as ModelRef,
    )
    from forgeai.models.registry import (
        ModelRegistry as ModelRegistry,
    )
    from forgeai.models.registry import (
        discover_model as discover_model,
    )
    from forgeai.models.registry import (
        is_multimodal as is_multimodal,
    )
    from forgeai.models.zoo import (
        MODEL_ALIASES as MODEL_ALIASES,
    )
    from forgeai.models.zoo import (
        resolve_model_name as resolve_model_name,
    )

_LAZY: dict[str, str] = {
    "CacheManager": "forgeai.models.loader",
    "download_model": "forgeai.models.loader",
    "get_cached_models": "forgeai.models.loader",
    "EngineSettings": "forgeai.models.manifest",
    "ForgeAIManifest": "forgeai.models.manifest",
    "GenerationDefaults": "forgeai.models.manifest",
    "KVCacheSettings": "forgeai.models.manifest",
    "ModelInfo": "forgeai.models.registry",
    "ModelRecord": "forgeai.models.registry",
    "ModelRef": "forgeai.models.registry",
    "ModelRegistry": "forgeai.models.registry",
    "discover_model": "forgeai.models.registry",
    "is_multimodal": "forgeai.models.registry",
    "MODEL_ALIASES": "forgeai.models.zoo",
    "resolve_model_name": "forgeai.models.zoo",
}

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


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
