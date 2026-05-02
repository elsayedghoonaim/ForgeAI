"""
forgeai batch — High-throughput offline extraction pipelines.

Bypasses API middleware to maximize GPU saturation for heavy tasks
like feeding thousands of PDFs through an OCR model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def batch(
    model: str = typer.Argument(..., help="Model name or repo ID"),
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSONL file"),
    output_file: str = typer.Option("output.jsonl", "--output", "-o", help="Output JSONL file"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per request"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    batch_size: int = typer.Option(32, "--batch-size", help="Requests per batch"),
    prompt_field: str = typer.Option("prompt", "--prompt-field", help="JSON field for prompt"),
    backend: str = typer.Option("auto", "--backend", "-b", help="Backend: auto, vllm, llama_cpp"),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", help="GPU layers for llama.cpp (-1 = all)"),
    n_ctx: int = typer.Option(4096, "--n-ctx", help="Context window for llama.cpp"),
) -> None:
    """High-throughput offline processing from JSONL files."""
    from forgeai.core.config import DevToolSettings
    from forgeai.core.telemetry import track_event
    from forgeai.models.zoo import resolve_model_name

    resolved = resolve_model_name(model)
    console.print("\n[bold cyan]ForgeAI Batch[/bold cyan]")
    console.print(f"  Model:  {resolved}")
    console.print(f"  Input:  {input_file}")
    console.print(f"  Output: {output_file}\n")

    track_event("command.batch", {"model": resolved})

    # Load input prompts
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]✗ Input file not found:[/red] {input_file}")
        raise typer.Exit(code=1)

    prompts = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    prompts.append(data.get(prompt_field, line))
                except json.JSONDecodeError:
                    prompts.append(line)

    console.print(f"  Loaded {len(prompts)} prompts")

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
            console.print(f"  1. Use the vllm backend instead:\n     forgeai batch {resolved} --backend vllm")
            console.print("  2. Convert it manually to GGUF using llama.cpp scripts.")
            raise typer.Exit(code=1)

    # Initialize engine
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

    # Process in batches
    results = []
    total_tokens = 0
    start_time = time.time()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=len(prompts))

            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]

                for prompt in batch_prompts:
                    try:
                        result = engine.generate(
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        results.append({
                            "prompt": prompt[:200],
                            "output": result.text,
                            "tokens": result.total_tokens,
                            "finish_reason": result.finish_reason,
                        })
                        total_tokens += result.total_tokens
                    except Exception as e:
                        results.append({"prompt": prompt[:200], "error": str(e)})

                    progress.advance(task)

        elapsed = time.time() - start_time

        # Write output
        with open(output_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        console.print("\n[green]✓ Batch complete[/green]")
        console.print(f"  Processed: {len(results)} prompts")
        console.print(f"  Tokens:    {total_tokens:,}")
        console.print(f"  Time:      {elapsed:.1f}s")
        console.print(f"  Speed:     {total_tokens / elapsed:.0f} tok/s" if elapsed > 0 else "")
        console.print(f"  Output:    {output_file}")
    finally:
        engine.shutdown()
