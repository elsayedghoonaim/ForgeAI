"""forgeai pull - model acquisition through the running daemon."""

from __future__ import annotations

import typer
from rich.console import Console

from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error
from forgeai.core.telemetry import track_event

console = Console()


def pull(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream progress events"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    """Download and cache a model through the running ForgeAI daemon.

    For private HuggingFace models, credentials must be configured in the daemon
    environment (e.g. HF_TOKEN or HUGGING_FACE_HUB_TOKEN).
    """

    track_event("command.pull", {"model": model, "stream": stream})

    console.print("\n[bold cyan]ForgeAI Pull[/bold cyan]")
    console.print(f"  Model: {model}\n")

    payload = {"name": model, "stream": stream}

    try:
        client = DaemonClient(host=host, port=port)
        if stream:
            success_observed = False
            for chunk in client.stream("POST", "/api/pull", json_data=payload):
                status = chunk.get("status")
                digest = chunk.get("digest")
                if status == "success":
                    success_observed = True

                if status:
                    if digest:
                        console.print(f"status: {status} digest: {digest}")
                    else:
                        console.print(f"status: {status}")
                elif "error" in chunk:
                    raise DaemonClientError(str(chunk["error"]))

            if not success_observed:
                raise DaemonClientError("Pull terminated prematurely without completing model acquisition.")

            console.print("\n[green]OK[/green] Model ready.")
        else:
            res = client.request("POST", "/api/pull", json_data=payload)
            status = res.get("status")
            digest = res.get("digest")

            if status != "success":
                err_msg = res.get("error") if isinstance(res, dict) else None
                raise DaemonClientError(f"Pull failed: {err_msg or status or 'Unknown error'}")

            if digest:
                console.print(f"status: {status} digest: {digest}")
            else:
                console.print(f"status: {status}")

            console.print("\n[green]OK[/green] Model ready.")
    except DaemonClientError as err:
        handle_cli_error(err)
