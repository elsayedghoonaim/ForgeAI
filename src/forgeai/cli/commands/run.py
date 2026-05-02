"""forgeai run - one-shot inference."""

from __future__ import annotations

import asyncio

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

console = Console()


async def _stream_once(
    engine,
    *,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    """Stream a single completion to the terminal."""

    async for chunk in engine.generate_stream(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    ):
        print(chunk, end="", flush=True, file=console.file)


def run(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Input prompt"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_p: float = typer.Option(0.95, "--top-p", help="Top-p sampling"),
    auto_optimize: bool = typer.Option(False, "--auto-optimize", help="Auto-tune tensor parallel size"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate without loading weights"),
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
    """Execute one-shot inference with VRAM estimation."""

    from forgeai.cli.runtime import print_runtime_tuning, resolve_runtime_tuning
    from forgeai.core.config import DevToolSettings
    from forgeai.core.telemetry import track_event
    from forgeai.models.zoo import resolve_model_name

    resolved = resolve_model_name(model)
    console.print("\n[bold cyan]ForgeAI Run[/bold cyan]")
    console.print(f"  Model: {resolved}")
    console.print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}\n")

    track_event("command.run", {"model": resolved, "dry_run": dry_run, "stream": stream})

    tuning = resolve_runtime_tuning(
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_utilization,
        auto_optimize=auto_optimize,
        run_mode=True,
    )
    print_runtime_tuning(tuning)

    if dry_run:
        console.print("[bold yellow]DRY RUN[/bold yellow] - validating without loading weights\n")
        try:
            from forgeai.utils.memory_estimator import estimate_from_preset, print_estimate

            estimate_kwargs: dict[str, float | int] = {
                "tensor_parallel_size": tuning.tensor_parallel_size,
            }
            if tuning.topology is not None and tuning.topology.gpu_count > 0:
                limiting_gpu = min(
                    tuning.topology.gpus[: tuning.tensor_parallel_size],
                    key=lambda gpu: gpu.free_memory_mb,
                )
                estimate_kwargs["available_vram_mb"] = limiting_gpu.free_memory_mb

            estimate = estimate_from_preset(resolved, **estimate_kwargs)
            if estimate:
                print_estimate(estimate, resolved)
            else:
                console.print("[dim]No preset for this model - skipping VRAM estimate[/dim]")
        except Exception as err:
            console.print(f"[yellow]WARN:[/yellow] estimation failed: {escape(str(err))}")
        console.print("\n[green]OK[/green] Dry run complete - profile is valid")
        return

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
            console.print(f"  1. Use the vllm backend instead:\n     forgeai run {resolved} --backend vllm")
            console.print("  2. Convert it manually to GGUF using llama.cpp scripts.")
            raise typer.Exit(code=1) from None

    settings = DevToolSettings(
        model_name=resolved,
        backend=backend,
        tensor_parallel_size=tuning.tensor_parallel_size,
        gpu_memory_utilization=tuning.gpu_memory_utilization,
        max_num_seqs=tuning.max_num_seqs or 256,
        max_model_len=tuning.max_model_len,
        max_num_batched_tokens=tuning.max_num_batched_tokens,
        enforce_eager=tuning.enforce_eager,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )

    try:
        from forgeai.cli.commands.chat import _LoadingSpinner
        from forgeai.core.engine import DevToolEngine

        engine = DevToolEngine(
            settings,
            streaming=stream,
            quiet_startup=not startup_logs,
        )
        if startup_logs:
            console.print("[dim]Waiting for engine readiness. The response appears after startup finishes.[/dim]")
            engine.initialize()
        else:
            with _LoadingSpinner(console):
                engine.initialize()
        console.print("[green]OK[/green] Engine ready.\n")

        try:
            if stream:
                console.print("[bold green]Output[/bold green]: ", end="")
                asyncio.run(
                    _stream_once(
                        engine,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                )
                print(file=console.file, flush=True)
                result = engine.last_result
                if result is None:
                    raise RuntimeError("Streaming completed without a final result.")
            else:
                result = engine.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                console.print(Panel(result.text, title="Output", border_style="green"))

            console.print(
                f"\n[dim]Tokens: {result.total_tokens} | "
                f"Speed: {result.tokens_per_second:.1f} tok/s | "
                f"Time: {result.elapsed_seconds:.2f}s[/dim]"
            )
        finally:
            engine.shutdown()
    except Exception as err:
        console.print(f"\n[red]ERROR:[/red] {escape(str(err))}")
        raise typer.Exit(code=1) from err
