"""
Runtime security enforcement.

- Blocks vLLM versions < 0.14.0 (CVE-2026-22807)
- Sanitizes file paths to prevent traversal attacks
- Validates tensor parallelism settings
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from rich.console import Console

console = Console()

# Minimum safe vLLM version
MIN_VLLM_VERSION = "0.14.0"


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string into a tuple of integers."""
    parts = re.findall(r"\d+", version_str)
    return tuple(int(p) for p in parts)


def check_vllm_version(
    min_version: str = MIN_VLLM_VERSION,
    *,
    announce_success: bool = True,
) -> bool:
    """
    Check that the installed vLLM version meets the minimum requirement.

    Raises RuntimeError if:
    - vLLM is installed but the version is below the minimum
    - The version cannot be determined

    Returns True if vLLM is not installed (allows dev mode without GPU).
    """
    try:
        import vllm

        installed = getattr(vllm, "__version__", None)
        if installed is None:
            raise RuntimeError(
                f"Cannot determine vLLM version. "
                f"Minimum required: {min_version} (CVE-2026-22807)"
            )

        if _parse_version(installed) < _parse_version(min_version):
            raise RuntimeError(
                f"SECURITY: vLLM {installed} is vulnerable to CVE-2026-22807.\n"
                f"Minimum required version: {min_version}\n"
                f"Upgrade with: pip install 'vllm>={min_version}'"
            )

        if announce_success:
            console.print(
                f"[green]✓[/green] vLLM version check passed: {installed} >= {min_version}"
            )
        return True

    except ImportError:
        # vLLM not installed — allow development mode
        return True


def sanitize_path(path: str, allowed_base: str | None = None) -> Path:
    """
    Sanitize a file path to prevent directory traversal attacks.

    Args:
        path: The path to sanitize.
        allowed_base: Optional base directory to restrict paths to.

    Returns:
        Resolved, safe Path object.

    Raises:
        ValueError: If the path contains traversal patterns or escapes the allowed base.
    """
    # Reject obvious traversal patterns
    if ".." in path:
        raise ValueError(f"Path traversal detected: {path!r}")

    resolved = Path(path).resolve()

    # If an allowed base is specified, ensure the resolved path falls within it
    if allowed_base:
        base = Path(allowed_base).resolve()
        if not str(resolved).startswith(str(base)):
            raise ValueError(
                f"Path escapes allowed directory.\n"
                f"  Path:    {resolved}\n"
                f"  Allowed: {base}"
            )

    return resolved


def validate_parallelism(tensor_parallel_size: int, available_gpus: int) -> int:
    """
    Validate tensor parallelism settings against available hardware.

    Args:
        tensor_parallel_size: Requested tensor parallel size.
        available_gpus: Number of available GPUs.

    Returns:
        Validated tensor parallel size.

    Raises:
        ValueError: If the configuration is invalid.
    """
    if tensor_parallel_size < 1:
        raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")

    if tensor_parallel_size > available_gpus:
        raise ValueError(
            f"tensor_parallel_size ({tensor_parallel_size}) exceeds "
            f"available GPUs ({available_gpus}). "
            f"Reduce to <= {available_gpus}."
        )

    # Warn about non-power-of-2 parallelism
    if tensor_parallel_size > 1 and (tensor_parallel_size & (tensor_parallel_size - 1)) != 0:
        console.print(
            f"[yellow]⚠[/yellow] tensor_parallel_size={tensor_parallel_size} is not a "
            f"power of 2. This may cause performance degradation."
        )

    return tensor_parallel_size


def validate_environment() -> dict[str, bool]:
    """
    Run a comprehensive environment security check.

    Returns a dict of check_name -> passed status.
    """
    results: dict[str, bool] = {}

    # Check vLLM version
    try:
        results["vllm_version_safe"] = check_vllm_version()
    except RuntimeError:
        results["vllm_version_safe"] = False

    # Check that cache directories are not world-writable
    cache_dir = os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )
    if os.path.exists(cache_dir):
        # On Unix, check permissions
        try:
            mode = os.stat(cache_dir).st_mode
            results["cache_dir_secure"] = not (mode & 0o002)  # Not world-writable
        except (OSError, AttributeError):
            results["cache_dir_secure"] = True  # Assume OK on Windows
    else:
        results["cache_dir_secure"] = True

    # Check for suspicious environment variables
    suspicious_vars = ["LD_PRELOAD", "PYTHONSTARTUP"]
    for var in suspicious_vars:
        if os.environ.get(var):
            results[f"no_suspicious_{var.lower()}"] = False
        else:
            results[f"no_suspicious_{var.lower()}"] = True

    return results
