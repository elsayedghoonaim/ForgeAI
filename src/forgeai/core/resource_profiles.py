"""
Resource profile catalog, platform policies, and selection rules.

Provides pure-data representations of KV-cache resource profiles, platform policy
enforcement, and deterministic selection helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnsupportedPlatformError(ValueError):
    """Raised when an inference platform is not supported by ForgeAI packaging."""

    pass


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

PRIMARY_PROFILES: tuple[str, ...] = ("auto", "fp8", "turboquant_4bit_nc", "turboquant_3bit_nc")

_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "auto",
    "bfloat16": "auto",
    "float16": "auto",
    "fp16": "auto",
}

VALID_WORKLOADS: set[str] = {"default", "latency", "quality", "memory"}


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


def _parse_cuda_compute_capability(
    cc: tuple[int, int] | float | str | None,
) -> tuple[int, int] | None:
    """Parse CUDA compute capability into a (major, minor) integer tuple."""
    if cc is None:
        return None
    if isinstance(cc, tuple) and len(cc) >= 2:
        return (int(cc[0]), int(cc[1]))
    if isinstance(cc, (float, int)):
        val = float(cc)
        major = int(val)
        minor = int(round((val - major) * 10))
        return (major, minor)
    if isinstance(cc, str):
        parts = cc.strip().split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))
        elif len(parts) == 1 and parts[0].isdigit():
            return (int(parts[0]), 0)
    return None


def select_resource_profile(
    *,
    platform: str = "cuda",
    cuda_compute_capability: tuple[int, int] | float | str | None = None,
    workload: str = "default",
    requested_profile: str | None = None,
    allow_fp8: bool = False,
    hw_validation_passed: bool = False,
    quality_validation_passed: bool = False,
    aggressive_quality_passed: bool = False,
) -> ResourceProfile:
    """
    Select a resource profile based on platform policy, hardware capabilities,
    workload requirements, and recorded validation evidence.

    Does not silently downgrade explicitly requested profiles.

    Raises:
        ValueError: If workload is unknown.
        UnsupportedPlatformError: If platform is unsupported (e.g. CPU, Metal, Intel).
        ProfileUnavailableError: If requested profile cannot be satisfied safely.
    """
    workload_norm = workload.lower().strip()
    if workload_norm not in VALID_WORKLOADS:
        raise ValueError(
            f"Unknown or unsupported workload '{workload}'. "
            f"Supported workloads: {', '.join(sorted(VALID_WORKLOADS))}"
        )

    plat_norm = platform.lower().strip()
    if plat_norm not in ("cuda", "rocm"):
        raise UnsupportedPlatformError(
            f"Platform '{platform}' is unsupported by ForgeAI packaging. "
            "Only NVIDIA CUDA and AMD ROCm inference are supported. CPU/Intel/Metal inference is unsupported."
        )

    parsed_cc = _parse_cuda_compute_capability(cuda_compute_capability)

    if requested_profile is not None and requested_profile.strip() != "":
        profile = get_resource_profile(requested_profile)

        # Check platform support
        if plat_norm not in profile.supported_platforms:
            raise UnsupportedPlatformError(
                f"Profile '{profile.name}' is only supported on {profile.supported_platforms}, "
                f"but running on '{platform}'."
            )

        # Check CUDA compute capability requirement if applicable
        if plat_norm == "cuda" and profile.min_cuda_compute_capability is not None:
            if parsed_cc is None or parsed_cc < profile.min_cuda_compute_capability:
                min_str = f"{profile.min_cuda_compute_capability[0]}.{profile.min_cuda_compute_capability[1]}"
                cc_str = f"{parsed_cc[0]}.{parsed_cc[1]}" if parsed_cc else "None"
                raise ProfileUnavailableError(
                    f"Profile '{profile.name}' requires NVIDIA CUDA compute capability >= {min_str}, "
                    f"but detected compute capability is {cc_str}."
                )

        # Check gating and validation requirements
        if profile.name == "turboquant_3bit_nc":
            if not (hw_validation_passed and quality_validation_passed and aggressive_quality_passed):
                raise ProfileUnavailableError(
                    "Profile 'turboquant_3bit_nc' is an aggressive POC-only profile and requires "
                    "hardware validation, quality validation, and aggressive quality gates to be explicitly passed."
                )

        if profile.name in ("turboquant_4bit_nc", "turboquant_k8v4"):
            if not (hw_validation_passed and quality_validation_passed):
                raise ProfileUnavailableError(
                    f"Profile '{profile.name}' requires both hardware and quality validation flags "
                    "to be explicitly recorded passed before it can be selected."
                )

        return profile

    # Default conservative auto-selection logic
    if workload_norm in ("latency", "quality", "default"):
        return RESOURCE_PROFILES["auto"]

    if workload_norm == "memory":
        if plat_norm == "rocm":
            if allow_fp8:
                return RESOURCE_PROFILES["fp8"]
            return RESOURCE_PROFILES["auto"]

        if plat_norm == "cuda":
            cc_ok = parsed_cc is not None and parsed_cc >= (7, 5)
            if cc_ok and hw_validation_passed and quality_validation_passed:
                return RESOURCE_PROFILES["turboquant_4bit_nc"]
            if allow_fp8:
                return RESOURCE_PROFILES["fp8"]
            return RESOURCE_PROFILES["auto"]

    return RESOURCE_PROFILES["auto"]
