"""forgeai chat - interactive terminal chat (a thin client over the running daemon)."""

from __future__ import annotations

import shutil
import threading
import time

import typer
from rich.console import Console

console = Console()
SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _startup_status_message(elapsed_seconds: float) -> str:
    """Return a concise startup status message for interactive chat."""

    if elapsed_seconds < 45:
        return (
            "Loading model and preparing the engine. First startup can take 1-3 minutes "
            "depending on model, GPU, and platform."
        )
    if elapsed_seconds < 120:
        return (
            "Still initializing. If the model is already in VRAM, the engine may still be "
            "compiling kernels and warming caches."
        )
    elapsed = int(elapsed_seconds)
    return (
        f"Still initializing after {elapsed}s. If this keeps going, check the daemon's "
        "logs (forgeai serve) for engine output."
    )


class _LoadingSpinner:
    """Simple TTY spinner that works consistently across older rich versions."""

    def __init__(self, console: Console) -> None:
        self._console = console
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._last_width = 0

    def __enter__(self) -> _LoadingSpinner:
        self._start_time = time.monotonic()
        if not self._console.is_terminal:
            self._console.print(f"[dim]{_startup_status_message(0.0)}[/dim]")
            return self

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if self._console.is_terminal and self._last_width:
            print(f"\r{' ' * self._last_width}\r", end="", file=self._console.file, flush=True)

    def _run(self) -> None:
        frame_index = 0
        while not self._stop.wait(0.1):
            elapsed = time.monotonic() - self._start_time
            message = _startup_status_message(elapsed)
            frame = SPINNER_FRAMES[frame_index % len(SPINNER_FRAMES)]
            frame_index += 1
            self._render_line(f"{frame} {message}")

    def _render_line(self, text: str) -> None:
        terminal_width = shutil.get_terminal_size(fallback=(100, 20)).columns
        max_width = max(20, terminal_width - 1)
        if len(text) > max_width:
            text = f"{text[: max_width - 1]}…"
        padding = max(0, self._last_width - len(text))
        print(
            f"\r{text}{' ' * padding}",
            end="",
            file=self._console.file,
            flush=True,
        )
        self._last_width = len(text)


def chat(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    system_prompt: str | None = typer.Option(None, "--system", help="Optional system prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p sampling"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens as they are generated"),
    keep_alive: str | None = typer.Option(None, "--keep-alive", help="Engine keep_alive TTL"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    """Start an interactive chat session through the running ForgeAI daemon.

    The model is loaded by the daemon (start it with ``forgeai serve``); Ctrl-C while a
    response is streaming cancels that response and returns to the prompt.
    """
    from forgeai.cli.repl import run_repl
    from forgeai.cli.runtime import DaemonClient, DaemonClientError, handle_cli_error
    from forgeai.core.telemetry import track_event

    if ".gguf" in model.lower():
        console.print(
            f"[red]ERROR:[/red] GGUF model format is unsupported in ForgeAI v2.0+ (model: {model!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )
        raise typer.Exit(code=1)

    console.print("\n[bold cyan]ForgeAI Chat[/bold cyan]")
    console.print(f"  Model: {model}")
    console.print("  Commands: /exit, /quit, /clear (Ctrl-C cancels a response)\n")

    track_event("command.chat", {"model": model, "stream": stream})

    try:
        client = DaemonClient(host=host, port=port)
        run_repl(
            client,
            model,
            system_prompt=system_prompt,
            options={"num_predict": max_tokens, "temperature": temperature, "top_p": top_p},
            keep_alive=keep_alive,
            stream=stream,
            prompt_label="You: ",
            wait_indicator=lambda: _LoadingSpinner(console),
        )
    except DaemonClientError as err:
        handle_cli_error(err)
