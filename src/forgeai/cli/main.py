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
from forgeai.cli.commands.create import create
from forgeai.cli.commands.doctor import doctor
from forgeai.cli.commands.ls import ls
from forgeai.cli.commands.ps import ps
from forgeai.cli.commands.pull import pull
from forgeai.cli.commands.rm import rm
from forgeai.cli.commands.run import run
from forgeai.cli.commands.serve import serve
from forgeai.cli.commands.show import show
from forgeai.cli.commands.stop import stop

console = Console()

app = typer.Typer(
    name=__app_name__,
    help="ForgeAI — A vLLM-powered CLI and API server for local LLMs.",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register direct commands (functions)
app.command("pull", help="Download and cache models through daemon (private repos require HF_TOKEN or HUGGING_FACE_HUB_TOKEN in daemon environment)")(pull)
app.command("run", help="Execute inference through the running daemon")(run)
app.command("chat", help="Interactive terminal chat")(chat)
app.command("serve", help="Start the Ollama-compatible daemon server")(serve)
app.command("ls", help="List registered local models")(ls)
app.command("ps", help="List active running model engines")(ps)
app.command("show", help="Display model metadata and manifest")(show)
app.command("rm", help="Remove a registered model tag")(rm)
app.command("stop", help="Stop a running model engine")(stop)
app.command("create", help="Create model tag from ForgeAI YAML manifest")(create)
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
    """ForgeAI — high-performance local LLM inference powered by vLLM."""
    if verbose:
        from forgeai.monitoring.logging import setup_logging
        setup_logging(level="DEBUG")
