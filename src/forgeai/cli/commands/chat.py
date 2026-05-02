"""forgeai chat - interactive terminal chat."""

from __future__ import annotations

import asyncio
import os
import shutil
import threading
import time

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

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
        f"Still initializing after {elapsed}s. If this keeps going, rerun with "
        "--startup-logs to inspect engine output."
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


def _print_startup_profile(
    *,
    model: str,
    settings,
    stream: bool,
    startup_logs: bool,
) -> None:
    """Print a concise chat startup summary."""

    console.print("[bold]Startup Profile[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  Mode: {'streaming chat' if stream else 'chat'}")
    console.print(f"  Tensor parallel: {settings.tensor_parallel_size}")
    console.print(f"  GPU util: {settings.gpu_memory_utilization:.2f}")
    console.print(f"  max_num_seqs: {settings.max_num_seqs}")
    console.print(f"  Execution: {'eager' if settings.enforce_eager else 'compiled'}")
    console.print(f"  max_model_len: {settings.max_model_len or 'model default'}")
    if startup_logs:
        console.print("[dim]Raw startup logs enabled.[/dim]\n")
        return

    if not os.environ.get("HF_TOKEN"):
        console.print("[dim]HF_TOKEN not set; cached models still work, uncached downloads may be slower.[/dim]")
    console.print("[dim]Raw startup logs suppressed. Use --startup-logs to show engine logs.[/dim]\n")


async def _read_user_input() -> str:
    """Prompt for user input without breaking the streaming event loop."""

    return (await asyncio.to_thread(Prompt.ask, "[bold blue]You[/bold blue]")).strip()


async def _stream_chat_session(
    engine,
    *,
    history: list[dict[str, str]],
    system_prompt: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    """Run the interactive chat loop with incremental token streaming."""

    while True:
        try:
            user_input = await _read_user_input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Chat ended.[/yellow]")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"/exit", "/quit"}:
            console.print("[yellow]Chat ended.[/yellow]")
            break
        if command == "/clear":
            history[:] = []
            if system_prompt:
                history.append({"role": "system", "content": system_prompt})
            console.print("[yellow]Chat history cleared.[/yellow]")
            continue

        history.append({"role": "user", "content": user_input})
        prompt_text = engine.build_prompt(history)

        console.print("[bold green]Assistant[/bold green]: ", end="")
        async for chunk in engine.generate_stream(
            prompt=prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            print(chunk, end="", flush=True)
        print()

        result = engine.last_result
        assistant_text = (result.text.strip() if result is not None else "") or "(empty response)"
        if result is not None:
            console.print(
                f"[dim]Tokens: {result.total_tokens} | "
                f"Speed: {result.tokens_per_second:.1f} tok/s | "
                f"Time: {result.elapsed_seconds:.2f}s[/dim]\n"
            )
        else:
            console.print()
        history.append({"role": "assistant", "content": assistant_text})


def chat(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    system_prompt: str | None = typer.Option(
        None,
        "--system",
        help="Optional system prompt",
    ),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p sampling"),
    auto_optimize: bool = typer.Option(False, "--auto-optimize", help="Auto-tune tensor parallel size"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: auto, vllm, llama_cpp"),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", help="GPU layers for llama.cpp (-1 = all)"),
    n_ctx: int = typer.Option(4096, "--n-ctx", help="Context window for llama.cpp"),
    gpu_utilization: float | None = typer.Option(
        None,
        "--gpu-util",
        help="GPU memory utilization (vLLM only, auto-tuned when omitted)",
    ),
    tensor_parallel: int | None = typer.Option(None, "--tp", help="Tensor parallel size (vLLM only)"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream tokens as they are generated"),
    startup_logs: bool = typer.Option(False, "--startup-logs", help="Show raw vLLM/HF startup logs"),
) -> None:
    """Start an interactive chat session in the terminal."""

    from forgeai.cli.runtime import print_runtime_tuning, resolve_runtime_tuning
    from forgeai.core.config import DevToolSettings
    from forgeai.core.telemetry import track_event
    from forgeai.models.zoo import resolve_model_name

    resolved = resolve_model_name(model)
    console.print("\n[bold cyan]ForgeAI Chat[/bold cyan]")
    console.print(f"  Model: {resolved}")
    console.print("  Commands: /exit, /quit, /clear\n")

    track_event("command.chat", {"model": resolved, "stream": stream})

    tuning = resolve_runtime_tuning(
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_utilization,
        auto_optimize=auto_optimize,
        chat_mode=True,
    )
    print_runtime_tuning(tuning)

    if backend in ("llama_cpp", "auto") and not resolved.endswith(".gguf") and backend == "llama_cpp":
        from forgeai.models.gguf_finder import find_gguf_for_model
        console.print(f"[dim]Searching HuggingFace for GGUF variants of {resolved}...[/dim]")
        candidates = find_gguf_for_model(resolved)
        if candidates:
            best = candidates[0]
            console.print(f"[dim]Found GGUF variant: {best.repo_id} / {best.filename}[/dim]")
            try:
                from huggingface_hub import hf_hub_download
                console.print(f"[dim]Downloading {best.filename} (this may take a while)...[/dim]")
                resolved = hf_hub_download(repo_id=best.repo_id, filename=best.filename)
                console.print(f"[green]OK[/green] Downloaded to {resolved}")
            except Exception as e:
                console.print(f"[red]ERROR:[/red] Failed to download GGUF: {e}")
                raise typer.Exit(code=1) from e
        else:
            console.print(f"\n[red]ERROR:[/red] Model \"{resolved}\" is not in GGUF format, and no GGUF variants were found.")
            console.print("\nThe llama_cpp backend requires GGUF models. Options:")
            console.print(f"  1. Use the vllm backend instead:\n     forgeai chat {resolved} --backend vllm")
            console.print("  2. Convert it manually to GGUF using llama.cpp scripts.")
            raise typer.Exit(code=1) from None

    settings_kwargs: dict[str, object] = {
        "model_name": resolved,
        "backend": backend,
        "tensor_parallel_size": tuning.tensor_parallel_size,
        "gpu_memory_utilization": tuning.gpu_memory_utilization,
        "enforce_eager": tuning.enforce_eager,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
    }
    if tuning.max_num_seqs is not None:
        settings_kwargs["max_num_seqs"] = tuning.max_num_seqs
    if tuning.max_model_len is not None:
        settings_kwargs["max_model_len"] = tuning.max_model_len
    settings = DevToolSettings(**settings_kwargs)
    _print_startup_profile(
        model=resolved,
        settings=settings,
        stream=stream,
        startup_logs=startup_logs,
    )

    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    try:
        from forgeai.core.engine import DevToolEngine

        engine = DevToolEngine(
            settings,
            streaming=stream,
            quiet_startup=not startup_logs,
        )
        if startup_logs:
            console.print(
                "[dim]Waiting for engine readiness. The chat prompt appears after startup finishes.[/dim]"
            )
            engine.initialize()
            console.print("[green]OK[/green] Chat engine ready.\n")
        else:
            with _LoadingSpinner(console):
                engine.initialize()
            console.print("[green]OK[/green] Chat engine ready.\n")
        try:
            if stream:
                asyncio.run(
                    _stream_chat_session(
                        engine,
                        history=history,
                        system_prompt=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
            else:
                while True:
                    try:
                        user_input = Prompt.ask("[bold blue]You[/bold blue]").strip()
                    except (EOFError, KeyboardInterrupt):
                        console.print("\n[yellow]Chat ended.[/yellow]")
                        break

                    if not user_input:
                        continue

                    command = user_input.lower()
                    if command in {"/exit", "/quit"}:
                        console.print("[yellow]Chat ended.[/yellow]")
                        break
                    if command == "/clear":
                        history = []
                        if system_prompt:
                            history.append({"role": "system", "content": system_prompt})
                        console.print("[yellow]Chat history cleared.[/yellow]")
                        continue

                    history.append({"role": "user", "content": user_input})
                    prompt_text = engine.build_prompt(history)
                    result = engine.generate(
                        prompt=prompt_text,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    assistant_text = result.text.strip() or "(empty response)"
                    console.print(Panel(assistant_text, title="Assistant", border_style="green"))
                    console.print(
                        f"[dim]Tokens: {result.total_tokens} | "
                        f"Speed: {result.tokens_per_second:.1f} tok/s | "
                        f"Time: {result.elapsed_seconds:.2f}s[/dim]\n"
                    )
                    history.append({"role": "assistant", "content": assistant_text})
        finally:
            engine.shutdown()
    except Exception as err:
        console.print(f"\n[red]ERROR:[/red] {escape(str(err))}")
        raise typer.Exit(code=1) from err
