"""
forgeai doctor — Automated diagnostics and deployment audit reports.
"""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def doctor(
    full: bool = typer.Option(False, "--full", help="Run full audit report"),
) -> None:
    """System diagnostics with actionable remediation and deployment audit report."""
    console.print("\n[bold cyan]ForgeAI Doctor[/bold cyan]\n")
    from forgeai.core.telemetry import track_event
    track_event("command.doctor")

    checks: list[tuple[str, bool, str]] = []

    # 1. Python version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    checks.append(("Python >= 3.10", py_ok, f"Installed: {py_ver}" + ("" if py_ok else " → Upgrade Python")))

    # 2. Backends
    # vLLM
    try:
        import vllm
        ver = getattr(vllm, "__version__", "unknown")
        from forgeai.core.security import MIN_VLLM_VERSION, _parse_version
        safe = _parse_version(ver) >= _parse_version(MIN_VLLM_VERSION)
        checks.append(
            (
                "vLLM >= 0.14.0",
                safe,
                escape(f"Installed: {ver}")
                + ("" if safe else escape(f" → pip install 'vllm>={MIN_VLLM_VERSION}'")),
            )
        )
    except ImportError:
        checks.append(
            (
                "vLLM available",
                False,
                "Not installed (optional)",
            )
        )

    # llama-cpp-python
    try:
        import llama_cpp
        ver = getattr(llama_cpp, "__version__", "unknown")
        checks.append(("llama-cpp-python available", True, escape(f"Installed: {ver}")))
    except ImportError:
        checks.append(("llama-cpp-python available", False, "Not installed (optional)"))

    # 3. GPU
    try:
        from forgeai.utils.gpu import detect_gpus
        topo = detect_gpus()
        gpu_ok = topo.gpu_count > 0
        checks.append(("GPU available", gpu_ok, f"{topo.gpu_count} GPU(s)" + (f" — {topo.gpus[0].name}" if gpu_ok else " → Check NVIDIA drivers")))
    except Exception:
        checks.append(("GPU available", False, "Detection failed → Install nvidia-ml-py"))

    # 4. CUDA
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    checks.append(("CUDA configured", bool(cuda_home), f"CUDA_HOME={cuda_home}" if cuda_home else "Not set → Set CUDA_HOME"))

    # 5. Dependencies
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

    # 6. Security checks
    try:
        from forgeai.core.security import validate_environment
        sec_results = validate_environment()
        for name, check_passed in sec_results.items():
            checks.append(
                (f"Security: {name}", check_passed, "Passed" if check_passed else "FAILED")
            )
    except Exception:
        checks.append(("Security checks", False, "Could not run security validation"))

    # Display results
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


def _print_audit_report(checks, score):
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
