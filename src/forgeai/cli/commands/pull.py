"""forgeai pull - model acquisition with integrated safety scanning."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape

console = Console()


def pull(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    cache_dir: str | None = typer.Option(None, "--cache-dir", help="Cache directory"),
    revision: str | None = typer.Option(None, "--revision", help="Model revision/branch"),
    token: str | None = typer.Option(
        None,
        "--token",
        help="HuggingFace API token (overrides stored token)",
    ),
    skip_scan: bool = typer.Option(False, "--skip-scan", help="Skip safety scanning"),
) -> None:
    """Download and cache a model with a default post-download safety scan."""

    from forgeai.core.telemetry import track_event
    from forgeai.models.loader import download_model
    from forgeai.models.zoo import resolve_model_name

    if token is None:
        from forgeai.cli.commands.config import get_stored_token

        token = get_stored_token()
        if token:
            console.print("[dim]Using stored HuggingFace token[/dim]")

    resolved = resolve_model_name(model)
    console.print("\n[bold cyan]vLLM DevTool Pull[/bold cyan]")
    console.print(f"  Model: {resolved}\n")

    track_event("command.pull", {"model": resolved})

    try:
        local_path = download_model(
            repo_id=resolved,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            enable_safety_scan=not skip_scan,
        )
        console.print(f"\n[green]OK[/green] Model ready: {local_path}")
    except Exception as err:
        console.print(f"\n[red]ERROR:[/red] pull failed: {escape(str(err))}")
        raise typer.Exit(code=1) from err
