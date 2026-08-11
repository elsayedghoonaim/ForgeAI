"""
forgeai batch — High-throughput offline extraction pipelines.

Bypasses API middleware to maximize GPU saturation for heavy tasks
like feeding thousands of PDFs through an OCR model.
"""

from __future__ import annotations

import asyncio
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
) -> None:
    """High-throughput offline processing from JSONL files."""
    from forgeai.core.config import DevToolSettings
    from forgeai.core.telemetry import track_event
    from forgeai.models.zoo import resolve_model_name

    resolved = resolve_model_name(model)
    if resolved.lower().endswith(".gguf") or ".gguf" in resolved.lower():
        console.print(
            f"[red]ERROR:[/red] GGUF model format is unsupported in ForgeAI v2.0+ (model: {resolved!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )
        raise typer.Exit(code=1)

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

    # Initialize engine
    settings = DevToolSettings(model_name=resolved)

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
                        result = asyncio.run(
                            engine.generate(
                                prompt=prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                            )
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
