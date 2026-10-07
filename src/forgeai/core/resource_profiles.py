"""
Resource profile catalog and lookup.

Provides pure-data representations of KV-cache resource profiles.
"""

from __future__ import annotations

from dataclasses import dataclass

from forgeai.core.config import RUNTIME_KV_CACHE_DTYPES


class ProfileUnavailableError(ValueError):
    """Raised when a requested resource profile is unavailable or unvalidated."""

    pass


@dataclass(frozen=True)
class ResourceProfile:
    """Immutable representation of a KV-cache resource profile."""

    name: str
    kv_cache_dtype: str
    expected_capacity_ratio: float
    bytes_per_element: float
    is_experimental: bool
    is_poc_gated: bool
    requires_validation: bool
    risk_level: str
    description: str
    supported_platforms: tuple[str, ...] = ("cuda", "rocm")
    min_cuda_compute_capability: tuple[int, int] | None = None


# Primary resource profiles and facts
RESOURCE_PROFILES: dict[str, ResourceProfile] = {
    "auto": ResourceProfile(
        name="auto",
        kv_cache_dtype="auto",
        expected_capacity_ratio=1.0,
        bytes_per_element=2.0,
        is_experimental=False,
        is_poc_gated=False,
        requires_validation=False,
        risk_level="lowest",
        description="BF16 baseline (auto). Production baseline with lowest risk.",
        supported_platforms=("cuda", "rocm"),
        min_cuda_compute_capability=None,
    ),
    "fp8": ResourceProfile(
        name="fp8",
        kv_cache_dtype="fp8",
        expected_capacity_ratio=2.0,
        bytes_per_element=1.0,
        is_experimental=False,
        is_poc_gated=False,
        requires_validation=True,
        risk_level="moderate",
        description="FP8 baseline. Expected ~2.0x capacity ratio. Platform and quality validation required.",
        supported_platforms=("cuda", "rocm"),
        min_cuda_compute_capability=None,
    ),
    "turboquant_4bit_nc": ResourceProfile(
        name="turboquant_4bit_nc",
        kv_cache_dtype="turboquant_4bit_nc",
        expected_capacity_ratio=3.8,
        bytes_per_element=2.0 / 3.8,
        is_experimental=True,
        is_poc_gated=True,
        requires_validation=True,
        risk_level="experimental",
        description="TurboQuant 4-bit NC. Target ~3.8x capacity ratio. Experimental & POC-gated.",
        supported_platforms=("cuda",),
        min_cuda_compute_capability=(7, 5),
    ),
    "turboquant_3bit_nc": ResourceProfile(
        name="turboquant_3bit_nc",
        kv_cache_dtype="turboquant_3bit_nc",
        expected_capacity_ratio=4.9,
        bytes_per_element=2.0 / 4.9,
        is_experimental=True,
        is_poc_gated=True,
        requires_validation=True,
        risk_level="high",
        description="Aggressive TurboQuant 3-bit NC. Target ~4.9x capacity ratio. POC-harness-only / disabled in normal runtime.",
        supported_platforms=("cuda",),
        min_cuda_compute_capability=(7, 5),
    ),
    "turboquant_k8v4": ResourceProfile(
        name="turboquant_k8v4",
        kv_cache_dtype="turboquant_k8v4",
        expected_capacity_ratio=2.0 / 0.75,  # ~2.67x
        bytes_per_element=0.75,
        is_experimental=True,
        is_poc_gated=True,
        requires_validation=True,
        risk_level="experimental",
        description="Native TurboQuant K8V4. Experimental & POC-gated.",
        supported_platforms=("cuda",),
        min_cuda_compute_capability=(7, 5),
    ),
}

# Profiles accepted by runtime config (single source of truth lives in core.config).
PRIMARY_PROFILES: tuple[str, ...] = RUNTIME_KV_CACHE_DTYPES

_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "auto",
    "bfloat16": "auto",
    "float16": "auto",
    "fp16": "auto",
}

def get_resource_profile(name_or_dtype: str) -> ResourceProfile:
    """
    Look up a resource profile by name or dtype alias.

    Raises:
        ProfileUnavailableError: If the profile name/dtype is unknown.
    """
    key = name_or_dtype.lower().strip()
    resolved = _DTYPE_ALIASES.get(key, key)
    profile = RESOURCE_PROFILES.get(resolved)
    if profile is None:
        valid_opts = sorted(set(RESOURCE_PROFILES.keys()).union(_DTYPE_ALIASES.keys()))
        raise ProfileUnavailableError(
            f"Unknown or unsupported resource profile '{name_or_dtype}'. "
            f"Supported profiles: {', '.join(valid_opts)}"
        )
    return profile

