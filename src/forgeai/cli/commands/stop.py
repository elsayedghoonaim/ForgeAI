"""
forgeai stop — Stop a running model engine in local daemon.
"""

from __future__ import annotations

import typer
from rich.console import Console

from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def stop(
    model: str = typer.Argument(..., help="Model tag name to stop"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    try:
        client = DaemonClient(host=host, port=port)
        client.request(
            "POST",
            "/api/generate",
            json_data={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        )
    except DaemonClientError as err:
        handle_cli_error(err)

    console.print(f"stopped '{model}'")
