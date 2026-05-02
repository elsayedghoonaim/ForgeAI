"""
Typer CLI root application.

Registers all sub-commands, version callback, and global flags.
"""

from __future__ import annotations

import typer
from rich.console import Console

from forgeai import __app_name__, __version__
from forgeai.cli.commands import (
    config,
    profile,
)
from forgeai.cli.commands.batch import batch
from forgeai.cli.commands.benchmark import benchmark
from forgeai.cli.commands.chat import chat
from forgeai.cli.commands.doctor import doctor
from forgeai.cli.commands.ps import ps
from forgeai.cli.commands.pull import pull
from forgeai.cli.commands.run import run
from forgeai.cli.commands.serve import serve

console = Console()

app = typer.Typer(
    name=__app_name__,
    help="ForgeAI — A unified, dual-backend CLI and API server for local LLMs.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register direct commands (functions)
app.command("pull", help="Download and cache models with safety scanning")(pull)
app.command("run", help="Execute one-shot inference")(run)
app.command("chat", help="Interactive terminal chat")(chat)
app.command("serve", help="Start the production API server")(serve)
app.command("ps", help="List active engines and GPU usage")(ps)
app.command("doctor", help="System diagnostics and audit reports")(doctor)
app.command("batch", help="High-throughput offline processing")(batch)
app.command("benchmark", help="Performance profiling")(benchmark)

# Register sub-command groups (Typer apps with subcommands)
app.add_typer(profile.app, name="profile", help="Manage deployment profiles")
app.add_typer(config.app, name="config", help="Manage persistent settings (HF token, etc.)")


def version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]{__app_name__}[/bold] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool | None = typer.Option(
        None, "--version", "-V",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Enable verbose output.",
    ),
) -> None:
    """ForgeAI — bridging raw performance and production usability across vLLM and llama.cpp."""
    if verbose:
        from forgeai.monitoring.logging import setup_logging
        setup_logging(level="DEBUG")
