"""
forgeai doctor — Automated diagnostics and deployment audit reports.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(invoke_without_command=True)


def check_python_version(sys_version_info: tuple[int, ...] = sys.version_info) -> tuple[bool, str]:
    """Check Python version requirement: >= 3.12, < 3.13 (Python 3.12)."""
    py_ver = f"{sys_version_info[0]}.{sys_version_info[1]}.{sys_version_info[2]}"
    py_ok = (3, 12) <= (sys_version_info[0], sys_version_info[1]) < (3, 13)
    detail = f"Installed: {py_ver}" if py_ok else f"Installed: {py_ver} → Upgrade to Python 3.12"
    return py_ok, detail


def check_vllm_diagnostic(vllm_module: Any = None) -> tuple[bool, str]:
    """
    Check exact vLLM version requirement (0.22.1).
    Dependency-injectable via vllm_module.
    """
    if vllm_module is None:
        try:
            import vllm

            vllm_module = vllm
        except ImportError:
            return False, "Not installed → pip install 'vllm==0.22.1'"

    if vllm_module is False:
        return False, "Not installed → pip install 'vllm==0.22.1'"

    ver = getattr(vllm_module, "__version__", None)
    if ver is None:
        return False, "Installed: unknown → pip install 'vllm==0.22.1'"

    try:
        from forgeai.core.security import REQUIRED_VLLM_VERSION, _parse_version

        installed_ver = _parse_version(str(ver))
        req_ver = _parse_version(REQUIRED_VLLM_VERSION)
        if installed_ver.public != req_ver.public:
            return False, escape(f"Installed: {ver} → pip install 'vllm=={REQUIRED_VLLM_VERSION}'")
    except Exception:
        return False, escape(f"Installed: {ver} → pip install 'vllm==0.22.1'")

    return True, escape(f"Installed: {ver}")


def check_platform_diagnostic(
    system_name: str | None = None,
    release_str: str | None = None,
    is_wsl: bool | None = None,
) -> tuple[bool, str]:
    """
    Check runtime OS platform support (Linux / WSL2 required).
    Native Windows, macOS (Metal), and Intel/CPU inference are unsupported.
    """
    sys_name = system_name if system_name is not None else platform.system()
    rel_str = release_str if release_str is not None else platform.release()

    if is_wsl is None:
        is_wsl = sys_name == "Linux" and (
            "wsl" in rel_str.lower()
            or "microsoft" in rel_str.lower()
            or os.environ.get("WSL_DISTRO_NAME") is not None
            or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
        )

    if sys_name == "Linux":
        if is_wsl:
            return True, "Linux (WSL2 supported)"
        return True, "Linux (native supported)"
    else:
        return False, f"{sys_name} unsupported — Linux or WSL2 required for vLLM engine execution"


def check_gpu_and_turboquant_diagnostic(
    gpu_topology: Any = None,
    is_rocm: bool | None = None,
    environ: dict[str, str] | None = None,
) -> list[tuple[str, bool, str]]:
    """
    Check GPU availability and TurboQuant readiness.
    Dependency-injectable with gpu_topology, is_rocm, environ.
    Returns list of (check_name, status_bool, detail_str).
    """
    env = environ if environ is not None else os.environ

    if is_rocm is None:
        is_rocm = bool(
            env.get("ROCM_HOME")
            or env.get("HIP_VISIBLE_DEVICES")
            or env.get("FORGEAI_IS_ROCM") == "1"
        )

    results: list[tuple[str, bool, str]] = []

    if gpu_topology is None and not is_rocm:
        try:
            from forgeai.utils.gpu import detect_gpus

            gpu_topology = detect_gpus()
        except Exception:
            gpu_topology = None

    gpu_count = getattr(gpu_topology, "gpu_count", 0) if gpu_topology else 0
    gpus = getattr(gpu_topology, "gpus", []) if gpu_topology else []

    if is_rocm:
        gpu_ok = gpu_count > 0
        gpu_detail = f"{gpu_count} AMD ROCm GPU(s)" if gpu_ok else "0 AMD ROCm GPUs detected"
        results.append(("GPU available", gpu_ok, gpu_detail))
        results.append((
            "TurboQuant readiness",
            False,
            "Explicitly unavailable/deferred on AMD ROCm (requires NVIDIA CUDA CC >= 7.5)",
        ))
        return results

    # CUDA / standard path
    gpu_ok = gpu_count > 0
    if not gpu_ok:
        results.append(("GPU available", False, "0 GPUs detected → Check NVIDIA drivers"))
        results.append(("TurboQuant readiness", False, "Unsupported (no NVIDIA CUDA GPU detected)"))
        return results

    first_gpu_name = getattr(gpus[0], "name", "GPU") if gpus else "GPU"
    results.append(("GPU available", True, f"{gpu_count} GPU(s) — {first_gpu_name}"))

    # Compute capability check for TurboQuant readiness
    has_real_evidence = False
    cc_ok = False
    cc_detail = "Missing CC evidence → unknown/fail"

    if gpus and hasattr(gpus[0], "compute_capability"):
        cc = gpus[0].compute_capability
        if isinstance(cc, tuple) and len(cc) == 2 and cc != (0, 0):
            has_real_evidence = True
            major, minor = cc
            if (major, minor) >= (7, 5):
                cc_ok = True
                cc_detail = f"NVIDIA CC {major}.{minor} >= 7.5"
            else:
                cc_ok = False
                cc_detail = f"NVIDIA CC {major}.{minor} < 7.5 (requires >= 7.5)"

    if not has_real_evidence:
        cc_ok = False
        cc_detail = "GPU Compute Capability missing/unknown (real evidence required >= 7.5)"

    results.append(("TurboQuant readiness", cc_ok, cc_detail))
    return results


@app.callback(invoke_without_command=True)
def doctor(
    full: bool = typer.Option(False, "--full", help="Run full audit report"),
) -> None:
    """System diagnostics with actionable remediation and deployment audit report."""
    console.print("\n[bold cyan]ForgeAI Doctor[/bold cyan]\n")
    from forgeai.core.telemetry import track_event

    track_event("command.doctor")

    checks: list[tuple[str, bool, str]] = []

    # 1. Python version check (>= 3.12, < 3.13)
    py_ok, py_detail = check_python_version()
    checks.append(("Python >= 3.12, < 3.13", py_ok, py_detail))

    # 2. vLLM check (exact 0.22.1)
    vllm_ok, vllm_detail = check_vllm_diagnostic()
    checks.append(("vLLM == 0.22.1", vllm_ok, vllm_detail))

    # 3. Platform OS check (Linux / WSL2)
    plat_ok, plat_detail = check_platform_diagnostic()
    checks.append(("Platform (Linux / WSL2)", plat_ok, plat_detail))

    # 4. GPU & TurboQuant readiness check
    gpu_checks = check_gpu_and_turboquant_diagnostic()
    checks.extend(gpu_checks)

    # 5. CUDA environment configuration
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    checks.append((
        "CUDA configured",
        bool(cuda_home),
        f"CUDA_HOME={cuda_home}" if cuda_home else "Not set → Set CUDA_HOME",
    ))

    # 6. Core Dependencies
    dependency_checks = [
        ("typer", "typer"),
        ("fastapi", "fastapi"),
        ("pydantic", "pydantic"),
        ("rich", "rich"),
        ("pyyaml", "yaml"),
        ("httpx", "httpx"),
        ("prometheus_client", "prometheus_client"),
    ]
    for package_name, import_name in dependency_checks:
        try:
            __import__(import_name)
            checks.append((package_name, True, "OK"))
        except ImportError:
            checks.append((package_name, False, escape(f"Missing → pip install {package_name}")))

    # 7. Security checks
    try:
        from forgeai.core.security import validate_environment

        sec_results = validate_environment()
        for name, check_passed in sec_results.items():
            checks.append(
                (f"Security: {name}", check_passed, "Passed" if check_passed else "FAILED")
            )
    except Exception:
        checks.append(("Security checks", False, "Could not run security validation"))

    # Display results table
    table = Table(title="Diagnostic Results", show_lines=True)
    table.add_column("Check", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Details")

    passed_count = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)

    for name, ok, detail in checks:
        status = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(name, status, detail)

    console.print(table)

    # Score
    score = (passed_count / total * 100) if total > 0 else 0
    color = "green" if score >= 80 else "yellow" if score >= 50 else "red"
    console.print(
        f"\n[{color}]Score: {score:.0f}/100[/{color}] ({passed_count}/{total} checks passed)"
    )

    if full:
        _print_audit_report(checks, score)


def _print_audit_report(checks: list[tuple[str, bool, str]], score: float) -> None:
    """Generate full deployment audit report."""
    report = (
        f"[bold]Deployment Audit Report[/bold]\n"
        f"Date:     {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Platform: {platform.system()} {platform.release()}\n"
        f"Python:   {sys.version.split()[0]}\n"
        f"Score:    {score:.0f}/100\n\n"
        f"[bold]{'Security' if score >= 80 else 'Risk'} Assessment:[/bold]\n"
    )
    failed = [(n, d) for n, ok, d in checks if not ok]
    if failed:
        report += "\n".join(f"  [red]✗[/red] {n}: {d}" for n, d in failed)
    else:
        report += "  [green]All checks passed.[/green]"

    console.print(Panel(report, title="Audit Report", border_style="cyan"))
