"""
forgeai rm — Remove a registered model tag.
"""

from __future__ import annotations

import typer
from rich.console import Console

from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def rm(
    model: str = typer.Argument(..., help="Model tag name to remove"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    try:
        client = DaemonClient(host=host, port=port)
        client.request("DELETE", "/api/delete", json_data={"name": model})
    except DaemonClientError as err:
        handle_cli_error(err)

    console.print(f"deleted '{model}'")
