"""
forgeai ps — Process and resource monitoring.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def ps() -> None:
    """List active engines with real-time GPU memory consumption."""
    console.print("\n[bold cyan]vLLM DevTool Processes[/bold cyan]\n")

    # GPU info
    try:
        from forgeai.utils.gpu import detect_gpus, print_gpu_table
        topology = detect_gpus()
        if topology.gpus:
            print_gpu_table(topology)
        else:
            console.print("[yellow]No GPUs detected.[/yellow]")
    except Exception as e:
        console.print(f"[dim]GPU detection unavailable: {e}[/dim]")

    # Check for running vLLM processes
    console.print("\n[bold]Active Processes:[/bold]")
    try:
        import psutil
        found = False
        table = Table(show_lines=True)
        table.add_column("PID", style="cyan")
        table.add_column("Name")
        table.add_column("CPU %", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Command")

        for proc in psutil.process_iter(["pid", "name", "cmdline", "cpu_percent", "memory_info"]):
            try:
                cmd = " ".join(proc.info.get("cmdline") or [])
                if "vllm" in cmd.lower() or "forgeai" in cmd.lower():
                    mem = proc.info.get("memory_info")
                    mem_str = f"{mem.rss / (1024**2):.0f} MB" if mem else "N/A"
                    table.add_row(
                        str(proc.info["pid"]),
                        proc.info["name"] or "",
                        f"{proc.info.get('cpu_percent', 0):.1f}%",
                        mem_str,
                        cmd[:80],
                    )
                    found = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if found:
            console.print(table)
        else:
            console.print("[dim]No active vLLM processes found.[/dim]")
    except ImportError:
        console.print("[dim]psutil not installed — process listing unavailable[/dim]")
