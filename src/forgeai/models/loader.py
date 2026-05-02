"""
Model download management and caching.

Handles HuggingFace Hub downloads with progress tracking, local caching,
and a default post-download safety scan.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console

from forgeai.utils.helpers import format_bytes

console = Console()


def download_model(
    repo_id: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    enable_safety_scan: bool = True,
) -> str:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct').
        cache_dir: Local cache directory. Defaults to HF_HOME.
        revision: Specific model revision/branch.
        token: HuggingFace API token for private models.
        enable_safety_scan: Run safety scanner after download.

    Returns:
        Local path to the downloaded model.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as err:
        raise RuntimeError(
            "huggingface_hub is required for model downloads.\n"
            "Install with: pip install huggingface-hub"
        ) from err

    # Suppress the symlink warning on Windows, which also messes up the terminal output
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    cache_dir = cache_dir or os.environ.get(
        "HF_HOME", os.path.expanduser("~/.cache/huggingface")
    )

    console.print(f"\n[bold]Downloading:[/bold] {repo_id}")
    if revision:
        console.print(f"  Revision: {revision}")
    console.print(f"  Cache:    {cache_dir}\n")

    # Use HuggingFace Hub's native progress display
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        revision=revision,
        token=token,
        ignore_patterns=["*.md", "*.txt", "LICENSE*", ".git*"],
    )

    console.print(f"\n[green]✓[/green] Model saved to: {local_path}")

    # Trigger the default post-download safety scan
    if enable_safety_scan:
        console.print("\n[bold]Running safety scan...[/bold]")
        try:
            from forgeai.models.safety_scanner import scan_model_weights

            scan_result = scan_model_weights(local_path)
            if scan_result["safe"]:
                console.print("[green]✓[/green] Safety scan passed")
            else:
                console.print(
                    f"[red]✗[/red] Safety scan flagged issues:\n"
                    f"  {scan_result.get('reason', 'Unknown')}"
                )
        except Exception as e:
            console.print(f"[yellow]⚠ Safety scan skipped: {e}[/yellow]")

    return local_path


def get_cached_models(cache_dir: str | None = None) -> list[dict[str, str]]:
    """List all locally cached models."""
    cache_dir = cache_dir or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    cache_path = Path(cache_dir)
    models: list[dict[str, str]] = []

    if not cache_path.exists():
        return models

    for entry in cache_path.iterdir():
        if entry.is_dir() and entry.name.startswith("models--"):
            parts = entry.name.replace("models--", "").split("--")
            repo_id = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else parts[0]

            # Calculate size
            total_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())

            models.append({
                "repo_id": repo_id,
                "path": str(entry),
                "size": format_bytes(total_size),
            })

    return models


def delete_cached_model(repo_id: str, cache_dir: str | None = None) -> bool:
    """Delete a cached model."""
    import shutil

    cache_dir = cache_dir or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "hub",
    )
    cache_path = Path(cache_dir)
    dir_name = f"models--{repo_id.replace('/', '--')}"
    model_path = cache_path / dir_name

    if model_path.exists():
        shutil.rmtree(model_path)
        console.print(f"[green]✓[/green] Deleted cached model: {repo_id}")
        return True
    else:
        console.print(f"[yellow]⚠[/yellow] Model not found in cache: {repo_id}")
        return False
