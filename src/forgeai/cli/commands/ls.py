"""
forgeai ls — List registered local model tags.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error

console = Console()
app = typer.Typer(invoke_without_command=True)


def format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable size string."""
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_idx = 0
    while size >= 1024.0 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)} B"
    return f"{size:.1f} {units[unit_idx]}"


@app.callback(invoke_without_command=True)
def ls(
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    try:
        client = DaemonClient(host=host, port=port)
        data = client.request("GET", "/api/tags")
    except DaemonClientError as err:
        handle_cli_error(err)

    models = data.get("models", [])
    if not models:
        console.print("[dim]No models registered.[/dim]")
        return

    table = Table(title="Registered Models", show_lines=True)
    table.add_column("NAME", style="cyan", no_wrap=True, overflow="ignore")
    table.add_column("SIZE", justify="right")
    table.add_column("MODIFIED")
    table.add_column("FORMAT")
    table.add_column("QUANTIZATION")
    table.add_column("KV CACHE")

    for item in models:
        name = item.get("name", "")
        size = format_size(item.get("size", 0))
        modified = item.get("modified_at", "")
        details = item.get("details", {}) or {}
        fmt = details.get("format", "safetensors")
        quant = details.get("weight_quantization") or details.get("quantization_level", "none")
        kv_cache = details.get("kv_cache_dtype", "auto")

        table.add_row(name, size, modified, fmt, quant, kv_cache)

    console.print(table)
