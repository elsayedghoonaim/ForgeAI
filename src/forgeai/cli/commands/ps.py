"""
forgeai ps — List active running models from local daemon.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from forgeai.cli.commands.ls import format_size
from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def ps(
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    try:
        client = DaemonClient(host=host, port=port)
        data = client.request("GET", "/api/ps")
    except DaemonClientError as err:
        handle_cli_error(err)

    models = data.get("models", [])
    if not models:
        console.print("[dim]No active models running.[/dim]")
        return

    table = Table(title="Running Models", show_lines=True)
    table.add_column("NAME", style="cyan", no_wrap=True, overflow="ignore")
    table.add_column("EXPIRATION")
    table.add_column("VRAM", justify="right")
    table.add_column("FORMAT")
    table.add_column("QUANTIZATION")
    table.add_column("KV CACHE")

    for item in models:
        name = item.get("name", "")
        expires = item.get("expires_at", "N/A")
        vram_bytes = item.get("size_vram", 0)
        vram_str = format_size(vram_bytes) if vram_bytes > 0 else "Unknown"
        details = item.get("details", {}) or {}
        fmt = details.get("format", "safetensors")
        quant = details.get("weight_quantization") or details.get("quantization_level", "none")
        kv_cache = details.get("kv_cache_dtype", "auto")

        table.add_row(name, expires, vram_str, fmt, quant, kv_cache)

    console.print(table)
