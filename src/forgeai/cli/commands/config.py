"""
forgeai config — Persistent configuration management.

Allows one-time setup of HuggingFace token and other settings,
stored in ~/.forgeai/config.yaml.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()

CONFIG_DIR = Path.home() / ".forgeai"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def _load_config() -> dict:
    """Load persistent config from disk."""
    if not CONFIG_FILE.exists():
        return {}
    import yaml
    with open(CONFIG_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _save_config(data: dict) -> None:
    """Save config to disk."""
    import yaml
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_stored_token() -> str | None:
    """Get the stored HuggingFace token, if any."""
    config = _load_config()
    return config.get("hf_token")


@app.command("set")
def set_value(
    key: str = typer.Argument(..., help="Config key (e.g., hf_token)"),
    value: str = typer.Argument(..., help="Config value"),
) -> None:
    """Set a persistent configuration value."""
    config = _load_config()
    config[key] = value

    # Mask sensitive values in display
    display_value = value
    if "token" in key.lower():
        display_value = value[:6] + "..." + value[-4:] if len(value) > 10 else "****"

    _save_config(config)
    console.print(f"[green]✓[/green] Set [bold]{key}[/bold] = {display_value}")
    console.print(f"  Stored in: {CONFIG_FILE}")


@app.command("get")
def get_value(
    key: str = typer.Argument(..., help="Config key to retrieve"),
) -> None:
    """Get a configuration value."""
    config = _load_config()
    if key in config:
        value = config[key]
        if "token" in key.lower():
            value = value[:6] + "..." + value[-4:] if len(value) > 10 else "****"
        console.print(f"[bold]{key}[/bold] = {value}")
    else:
        console.print(f"[yellow]⚠[/yellow] Key not found: {key}")
        raise typer.Exit(code=1)


@app.command("list")
def list_config() -> None:
    """Show all configuration values."""
    config = _load_config()
    if not config:
        console.print("[dim]No configuration set yet.[/dim]")
        console.print("  Run: [cyan]forgeai config set hf_token YOUR_TOKEN[/cyan]")
        return

    table = Table(title="Configuration", show_lines=True)
    table.add_column("Key", style="cyan bold")
    table.add_column("Value", style="white")

    for key, value in config.items():
        display = value
        if "token" in key.lower() and isinstance(value, str):
            display = value[:6] + "..." + value[-4:] if len(value) > 10 else "****"
        table.add_row(key, str(display))

    console.print(table)
    console.print(f"\n[dim]Config file: {CONFIG_FILE}[/dim]")


@app.command("delete")
def delete_value(
    key: str = typer.Argument(..., help="Config key to remove"),
) -> None:
    """Remove a configuration value."""
    config = _load_config()
    if key in config:
        del config[key]
        _save_config(config)
        console.print(f"[green]✓[/green] Removed: {key}")
    else:
        console.print(f"[yellow]⚠[/yellow] Key not found: {key}")


@app.command("login")
def login(
    token: str = typer.Argument(..., help="HuggingFace API token"),
) -> None:
    """Quick login — store your HuggingFace token."""
    config = _load_config()
    config["hf_token"] = token
    _save_config(config)

    masked = token[:6] + "..." + token[-4:] if len(token) > 10 else "****"
    console.print(f"[green]✓[/green] HuggingFace token saved: {masked}")
    console.print("  Token will be used automatically for model downloads.")
    console.print(f"  Stored in: {CONFIG_FILE}")
