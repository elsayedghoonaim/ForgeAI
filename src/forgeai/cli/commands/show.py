"""
forgeai show — Display manifest and model details.
"""

from __future__ import annotations

import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def show(
    model: str = typer.Argument(..., help="Model tag name"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON response"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    try:
        client = DaemonClient(host=host, port=port)
        data = client.request("POST", "/api/show", json_data={"name": model})
    except DaemonClientError as err:
        handle_cli_error(err)

    if json_output:
        console.print_json(json.dumps(data))
        return

    modelfile = data.get("modelfile", "")
    parameters = data.get("parameters", "")
    template = data.get("template", "")
    system = data.get("system", "")
    details = data.get("details", {}) or {}
    model_info = data.get("model_info", {}) or {}

    console.print(f"\n[bold cyan]Model Details for '{model}'[/bold cyan]\n")

    if details:
        table = Table(title="Engine & Quantization Details", show_header=True)
        table.add_column("Property", style="bold")
        table.add_column("Value")
        for k, v in details.items():
            table.add_row(k, str(v))
        console.print(table)
        console.print()

    if model_info:
        info_table = Table(title="Model Info", show_header=True)
        info_table.add_column("Key", style="bold")
        info_table.add_column("Value")
        for k, v in model_info.items():
            info_table.add_row(k, str(v))
        console.print(info_table)
        console.print()

    if parameters:
        console.print(Panel(parameters, title="Parameters", border_style="blue"))

    if system:
        console.print(Panel(system, title="System Prompt", border_style="green"))

    if template:
        console.print(Panel(template, title="Chat Template", border_style="magenta"))

    if modelfile:
        console.print(Panel(modelfile, title="Manifest (Modelfile Equivalent)", border_style="yellow"))
