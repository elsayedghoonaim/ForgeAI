"""
General utility functions.

Pretty printing, file size formatting, YAML I/O, retry decorator, and more.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import yaml
from rich.console import Console

console = Console()
F = TypeVar("F", bound=Callable[..., Any])


def format_bytes(num_bytes: float) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:,.1f} PB"


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_number(n: float) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return f"{n:.0f}"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Save a dict to a YAML file, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return path


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator for retrying a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier for delay after each retry.
        exceptions: Tuple of exception types to catch.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        console.print(
                            f"[yellow]⚠ Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f}s...[/yellow]"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        console.print(
                            f"[red]✗ All {max_attempts} attempts failed.[/red]"
                        )

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


def truncate_string(s: str, max_length: int = 80, suffix: str = "...") -> str:
    """Truncate a string to max_length, adding suffix if truncated."""
    if len(s) <= max_length:
        return s
    return s[: max_length - len(suffix)] + suffix


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
