"""
forgeai profile — Reproducible deployment profiles (.yaml).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from forgeai.core.config import RUNTIME_KV_CACHE_DTYPES, normalize_kv_cache_dtype

console = Console()
app = typer.Typer()



@app.command("save")
def save_profile(
    name: str = typer.Argument(..., help="Profile name"),
    model: str = typer.Option("", "--model", help="Model name"),
    tp: int = typer.Option(1, "--tp", help="Tensor parallel size"),
    gpu_util: float = typer.Option(0.85, "--gpu-util", help="GPU utilization"),
    max_model_len: int | None = typer.Option(None, "--max-model-len", help="Max context length"),
    max_num_seqs: int = typer.Option(4, "--max-num-seqs", help="Max concurrent sequences"),
    host: str = typer.Option("127.0.0.1", "--host", help="Server host"),
    port: int = typer.Option(11434, "--port", help="Server port"),
    kv_cache_dtype: str = typer.Option(
        "auto",
        "--kv-cache-dtype",
        help=f"KV-cache dtype profile ({', '.join(RUNTIME_KV_CACHE_DTYPES)})",
    ),
) -> None:
    """Save a reproducible deployment profile."""
    from forgeai import __version__
    from forgeai.core.config import DevToolSettings
    from forgeai.utils.helpers import save_yaml

    try:
        kv_lower = normalize_kv_cache_dtype(kv_cache_dtype)
    except ValueError as err:
        console.print(f"[red]✗ Error:[/red] {err}")
        raise typer.Exit(code=1) from err

    settings = DevToolSettings()
    profile_dir = Path(settings.profiles_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    profile_data = {
        "name": name,
        "version": __version__,
        "model": {
            "name": model,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
        },
        "kv_cache": {
            "dtype": kv_lower,
        },
        "gpu": {"tensor_parallel_size": tp, "gpu_memory_utilization": gpu_util},
        "server": {"host": host, "port": port},
    }

    filepath = save_yaml(profile_data, profile_dir / f"{name}.yaml")
    console.print(f"[green]✓[/green] Profile saved: {filepath}")


@app.command("load")
def load_profile(
    name: str = typer.Argument(..., help="Profile name to load"),
) -> None:
    """Load and display a deployment profile."""
    from forgeai.core.config import DevToolSettings
    from forgeai.utils.helpers import load_yaml

    settings = DevToolSettings()
    filepath = Path(settings.profiles_dir) / f"{name}.yaml"

    if not filepath.exists():
        console.print(f"[red]✗ Profile not found:[/red] {filepath}")
        raise typer.Exit(code=1)

    data = load_yaml(filepath)
    console.print(f"\n[bold cyan]Profile: {name}[/bold cyan]")
    import yaml

    console.print(yaml.dump(data, default_flow_style=False))


@app.command("list")
def list_profiles() -> None:
    """List all saved deployment profiles."""
    from rich.table import Table

    from forgeai.core.config import DevToolSettings

    settings = DevToolSettings()
    profile_dir = Path(settings.profiles_dir)

    if not profile_dir.exists():
        console.print("[dim]No profiles directory found.[/dim]")
        return

    profiles = list(profile_dir.glob("*.yaml"))
    if not profiles:
        console.print("[dim]No profiles saved yet.[/dim]")
        return

    table = Table(title="Deployment Profiles", show_lines=True)
    table.add_column("Name", style="cyan bold")
    table.add_column("Size")
    table.add_column("Path")

    for p in sorted(profiles):
        from forgeai.utils.helpers import format_bytes

        table.add_row(p.stem, format_bytes(p.stat().st_size), str(p))

    console.print(table)


@app.command("delete")
def delete_profile(
    name: str = typer.Argument(..., help="Profile name to delete"),
) -> None:
    """Delete a saved deployment profile."""
    from forgeai.core.config import DevToolSettings

    settings = DevToolSettings()
    filepath = Path(settings.profiles_dir) / f"{name}.yaml"

    if filepath.exists():
        filepath.unlink()
        console.print(f"[green]✓[/green] Deleted profile: {name}")
    else:
        console.print(f"[yellow]⚠ Profile not found:[/yellow] {name}")
