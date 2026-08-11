"""
Core module — Engine lifecycle, configuration, security, and telemetry.
"""

from forgeai.core.config import DevToolSettings, KVCacheSettings
from forgeai.core.engine import (
    AdmissionVRAMError,
    DevToolEngine,
    EngineDrainingError,
    EngineKey,
    EngineLease,
    EngineLoadCancelledError,
    EngineManager,
    EngineManagerError,
    EngineOOMError,
    EngineQueueFullError,
    EngineShuttingDownError,
    EngineState,
    EngineStatus,
    KeepAlivePolicy,
    NoCompatibleGPUError,
    ROCmDeferredError,
    TurboQuantHWCCError,
    parse_keep_alive,
)
from forgeai.core.resource_profiles import (
    PRIMARY_PROFILES,
    RESOURCE_PROFILES,
    ProfileUnavailableError,
    ResourceProfile,
    UnsupportedPlatformError,
    get_resource_profile,
    select_resource_profile,
)
from forgeai.core.security import check_vllm_version, sanitize_path, validate_parallelism

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
    "UnsupportedPlatformError",
    "check_vllm_version",
    "get_resource_profile",
    "parse_keep_alive",
    "sanitize_path",
    "select_resource_profile",
    "validate_parallelism",
]
