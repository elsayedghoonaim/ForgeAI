"""Core module — Engine lifecycle, configuration, security, and telemetry.

Imports are lazy (PEP 562) so importing a submodule does not load the whole package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forgeai.core.config import (
        DevToolSettings as DevToolSettings,
    )
    from forgeai.core.config import (
        KVCacheSettings as KVCacheSettings,
    )
    from forgeai.core.engine import (
        AdmissionVRAMError as AdmissionVRAMError,
    )
    from forgeai.core.engine import (
        DevToolEngine as DevToolEngine,
    )
    from forgeai.core.engine import (
        EngineDrainingError as EngineDrainingError,
    )
    from forgeai.core.engine import (
        EngineKey as EngineKey,
    )
    from forgeai.core.engine import (
        EngineLease as EngineLease,
    )
    from forgeai.core.engine import (
        EngineLoadCancelledError as EngineLoadCancelledError,
    )
    from forgeai.core.engine import (
        EngineManager as EngineManager,
    )
    from forgeai.core.engine import (
        EngineManagerError as EngineManagerError,
    )
    from forgeai.core.engine import (
        EngineOOMError as EngineOOMError,
    )
    from forgeai.core.engine import (
        EngineQueueFullError as EngineQueueFullError,
    )
    from forgeai.core.engine import (
        EngineShuttingDownError as EngineShuttingDownError,
    )
    from forgeai.core.engine import (
        EngineState as EngineState,
    )
    from forgeai.core.engine import (
        EngineStatus as EngineStatus,
    )
    from forgeai.core.engine import (
        KeepAlivePolicy as KeepAlivePolicy,
    )
    from forgeai.core.engine import (
        NoCompatibleGPUError as NoCompatibleGPUError,
    )
    from forgeai.core.engine import (
        ROCmDeferredError as ROCmDeferredError,
    )
    from forgeai.core.engine import (
        TurboQuantHWCCError as TurboQuantHWCCError,
    )
    from forgeai.core.engine import (
        parse_keep_alive as parse_keep_alive,
    )
    from forgeai.core.resource_profiles import (
        PRIMARY_PROFILES as PRIMARY_PROFILES,
    )
    from forgeai.core.resource_profiles import (
        RESOURCE_PROFILES as RESOURCE_PROFILES,
    )
    from forgeai.core.resource_profiles import (
        ProfileUnavailableError as ProfileUnavailableError,
    )
    from forgeai.core.resource_profiles import (
        ResourceProfile as ResourceProfile,
    )
    from forgeai.core.resource_profiles import (
        get_resource_profile as get_resource_profile,
    )
    from forgeai.core.security import (
        check_vllm_version as check_vllm_version,
    )
    from forgeai.core.security import (
        sanitize_path as sanitize_path,
    )
    from forgeai.core.security import (
        validate_parallelism as validate_parallelism,
    )

_LAZY: dict[str, str] = {
    "DevToolSettings": "forgeai.core.config",
    "KVCacheSettings": "forgeai.core.config",
    "AdmissionVRAMError": "forgeai.core.engine",
    "DevToolEngine": "forgeai.core.engine",
    "EngineDrainingError": "forgeai.core.engine",
    "EngineKey": "forgeai.core.engine",
    "EngineLease": "forgeai.core.engine",
    "EngineLoadCancelledError": "forgeai.core.engine",
    "EngineManager": "forgeai.core.engine",
    "EngineManagerError": "forgeai.core.engine",
    "EngineOOMError": "forgeai.core.engine",
    "EngineQueueFullError": "forgeai.core.engine",
    "EngineShuttingDownError": "forgeai.core.engine",
    "EngineState": "forgeai.core.engine",
    "EngineStatus": "forgeai.core.engine",
    "KeepAlivePolicy": "forgeai.core.engine",
    "NoCompatibleGPUError": "forgeai.core.engine",
    "ROCmDeferredError": "forgeai.core.engine",
    "TurboQuantHWCCError": "forgeai.core.engine",
    "parse_keep_alive": "forgeai.core.engine",
    "PRIMARY_PROFILES": "forgeai.core.resource_profiles",
    "RESOURCE_PROFILES": "forgeai.core.resource_profiles",
    "ProfileUnavailableError": "forgeai.core.resource_profiles",
    "ResourceProfile": "forgeai.core.resource_profiles",
    "get_resource_profile": "forgeai.core.resource_profiles",
    "check_vllm_version": "forgeai.core.security",
    "sanitize_path": "forgeai.core.security",
    "validate_parallelism": "forgeai.core.security",
}

__all__ = [
    "AdmissionVRAMError",
    "DevToolEngine",
    "DevToolSettings",
    "EngineDrainingError",
    "EngineKey",
    "EngineLease",
    "EngineLoadCancelledError",
    "EngineManager",
    "EngineManagerError",
    "EngineOOMError",
    "EngineQueueFullError",
    "EngineShuttingDownError",
    "EngineState",
    "EngineStatus",
    "KVCacheSettings",
    "KeepAlivePolicy",
    "NoCompatibleGPUError",
    "PRIMARY_PROFILES",
    "ProfileUnavailableError",
    "RESOURCE_PROFILES",
    "ROCmDeferredError",
    "ResourceProfile",
    "TurboQuantHWCCError",
    "check_vllm_version",
    "get_resource_profile",
    "parse_keep_alive",
    "sanitize_path",
    "validate_parallelism",
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
