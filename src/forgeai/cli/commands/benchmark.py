"""
forgeai benchmark — Performance profiling and compute justification.

Standardized performance profiling to provide exact numbers
for management to justify compute purchases.
"""

from __future__ import annotations

import statistics

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()
app = typer.Typer(invoke_without_command=True)

# Standard benchmark prompts
BENCHMARK_PROMPTS = [
    "Explain the theory of general relativity in simple terms.",
    "Write a Python function that implements binary search.",
    "Summarize the key events of World War II in 200 words.",
    "What are the main differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis step by step.",
]


@app.callback(invoke_without_command=True)
def benchmark(
    model: str = typer.Argument(..., help="Model name or repo ID"),
    num_iterations: int = typer.Option(5, "--iterations", "-n", help="Number of iterations"),
    max_tokens: int = typer.Option(256, "--max-tokens", help="Max tokens per request"),
    warmup: int = typer.Option(1, "--warmup", help="Warmup iterations"),
    prompt: str | None = typer.Option(None, "--prompt", help="Custom benchmark prompt"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: auto, vllm, llama_cpp"),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", help="GPU layers for llama.cpp (-1 = all)"),
    n_ctx: int = typer.Option(4096, "--n-ctx", help="Context window for llama.cpp"),
) -> None:
    """Standardized performance benchmark for compute justification."""
    from forgeai.core.config import DevToolSettings
    from forgeai.core.telemetry import track_event
    from forgeai.models.zoo import resolve_model_name

    resolved = resolve_model_name(model)
    console.print("\n[bold cyan]ForgeAI Benchmark[/bold cyan]")
    console.print(f"  Model:      {resolved}")
    console.print(f"  Iterations: {num_iterations}")
    console.print(f"  Max tokens: {max_tokens}\n")

    track_event("command.benchmark", {"model": resolved})

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
                raise typer.Exit(code=1)
        else:
            console.print(f"\n[red]ERROR:[/red] Model \"{resolved}\" is not in GGUF format, and no GGUF variants were found.")
            console.print("\nThe llama_cpp backend requires GGUF models. Options:")
            console.print(f"  1. Use the vllm backend instead:\n     forgeai benchmark {resolved} --backend vllm")
            console.print("  2. Convert it manually to GGUF using llama.cpp scripts.")
            raise typer.Exit(code=1)

    settings = DevToolSettings(
        model_name=resolved,
        backend=backend,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
    )

    try:
        from forgeai.core.engine import DevToolEngine
        engine = DevToolEngine(settings)
        engine.initialize()
    except Exception as e:
        console.print(f"[red]✗ Engine init failed:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from e

    prompts = [prompt] * num_iterations if prompt else (BENCHMARK_PROMPTS * ((num_iterations // 5) + 1))[:num_iterations]

    # Warmup
    if warmup > 0:
        console.print(f"[dim]Warming up ({warmup} iterations)...[/dim]")
        for _ in range(warmup):
            engine.generate(prompts[0], max_tokens=max_tokens, temperature=0.0)

    # Benchmark
    latencies = []
    token_rates = []
    prompt_tokens_list = []
    completion_tokens_list = []

    try:
        console.print("[dim]Running benchmark...[/dim]\n")
        for i, p in enumerate(prompts):
            result = engine.generate(p, max_tokens=max_tokens, temperature=0.0)
            latencies.append(result.elapsed_seconds)
            if result.elapsed_seconds > 0:
                token_rates.append(result.completion_tokens / result.elapsed_seconds)
            prompt_tokens_list.append(result.prompt_tokens)
            completion_tokens_list.append(result.completion_tokens)
            console.print(f"  [{i+1}/{num_iterations}] {result.elapsed_seconds:.2f}s — {result.completion_tokens} tokens")
    finally:
        engine.shutdown()

    # Print report
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Model", resolved)
    table.add_row("Iterations", str(num_iterations))
    table.add_row("Avg Latency", f"{statistics.mean(latencies):.3f}s")
    table.add_row("P50 Latency", f"{statistics.median(latencies):.3f}s")
    if len(latencies) >= 2:
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        table.add_row("P95 Latency", f"{sorted_lat[min(p95_idx, len(sorted_lat)-1)]:.3f}s")
    table.add_row("Avg Tokens/s", f"{statistics.mean(token_rates):.1f}" if token_rates else "N/A")
    table.add_row("Total Prompt Tokens", f"{sum(prompt_tokens_list):,}")
    table.add_row("Total Completion Tokens", f"{sum(completion_tokens_list):,}")
    table.add_row("Total Time", f"{sum(latencies):.2f}s")

    console.print("\n")
    console.print(table)
