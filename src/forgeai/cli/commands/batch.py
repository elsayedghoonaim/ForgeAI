"""
forgeai batch — offline processing of a JSONL file through the running daemon.

A thin client: prompts are sent concurrently (up to ``--batch-size`` in flight) to the
daemon's ``/api/generate``, so vLLM's continuous batching does the work.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape

console = Console()
app = typer.Typer(invoke_without_command=True)


def load_prompts(path: Path, prompt_field: str) -> tuple[list[tuple[int, str]], list[str]]:
    """Parse a JSONL file into ``[(line_number, prompt)]`` plus a list of problems.

    Every non-empty line must be a JSON object whose ``prompt_field`` is a string; other
    lines are skipped and reported.
    """
    prompts: list[tuple[int, str]] = []
    problems: list[str] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as err:
                problems.append(f"line {lineno}: invalid JSON ({err.msg})")
                continue
            if not isinstance(data, dict):
                problems.append(f"line {lineno}: expected a JSON object, got {type(data).__name__}")
                continue
            value = data.get(prompt_field)
            if not isinstance(value, str):
                problems.append(f"line {lineno}: field {prompt_field!r} missing or not a string")
                continue
            prompts.append((lineno, value))
    return prompts, problems


async def process_prompts(
    http: Any,
    model: str,
    prompts: list[tuple[int, str]],
    *,
    max_tokens: int,
    temperature: float,
    batch_size: int,
    on_done: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    """Send prompts concurrently (at most ``batch_size`` in flight); results keep input order."""
    import asyncio

    import httpx

    semaphore = asyncio.Semaphore(max(1, batch_size))

    async def one(lineno: int, prompt: str) -> dict[str, Any]:
        record: dict[str, Any] = {"line": lineno, "prompt": prompt[:200]}
        async with semaphore:
            try:
                resp = await http.post(
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": max_tokens, "temperature": temperature},
                    },
                )
                if resp.status_code >= 400:
                    try:
                        detail = resp.json().get("error", resp.text)
                    except Exception:
                        detail = resp.text
                    record["error"] = f"HTTP {resp.status_code}: {str(detail).strip()}"
                else:
                    data = resp.json()
                    record["output"] = data.get("response", "")
                    record["tokens"] = int(data.get("prompt_eval_count") or 0) + int(
                        data.get("eval_count") or 0
                    )
                    record["finish_reason"] = data.get("done_reason", "stop")
            except httpx.TimeoutException:
                record["error"] = "daemon did not respond in time"
            except httpx.RequestError as err:
                record["error"] = f"request failed: {err}"
            except Exception as err:
                record["error"] = str(err)
        if on_done is not None:
            on_done()
        return record

    return list(await asyncio.gather(*(one(n, p) for n, p in prompts)))


@app.callback(invoke_without_command=True)
def batch(
    model: str = typer.Argument(..., help="Model name or repo ID"),
    input_file: str = typer.Option(..., "--input", "-i", help="Input JSONL file"),
    output_file: str = typer.Option("output.jsonl", "--output", "-o", help="Output JSONL file"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per request"),
    temperature: float = typer.Option(0.0, "--temperature", help="Sampling temperature"),
    batch_size: int = typer.Option(32, "--batch-size", help="Maximum concurrent requests"),
    prompt_field: str = typer.Option("prompt", "--prompt-field", help="JSON field for prompt"),
    host: str | None = typer.Option(None, "--host", help="Daemon host"),
    port: int | None = typer.Option(None, "--port", help="Daemon port"),
) -> None:
    """Process a JSONL file of prompts through the running daemon."""
    import asyncio

    import httpx
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from forgeai.cli.runtime import DaemonClient, DaemonClientError, exit_if_gguf, handle_cli_error
    from forgeai.core.telemetry import track_event

    exit_if_gguf(model)
    if batch_size < 1:
        handle_cli_error("--batch-size must be at least 1.")

    console.print("\n[bold cyan]ForgeAI Batch[/bold cyan]")
    console.print(f"  Model:  {model}")
    console.print(f"  Input:  {input_file}")
    console.print(f"  Output: {output_file}\n")

    track_event("command.batch", {"model": model})

    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]✗ Input file not found:[/red] {input_file}")
        raise typer.Exit(code=1)

    prompts, problems = load_prompts(input_path, prompt_field)
    console.print(f"  Loaded {len(prompts)} prompts")
    if problems:
        console.print(f"[yellow]  Skipped {len(problems)} invalid line(s):[/yellow]")
        for problem in problems[:20]:
            console.print(f"    {escape(problem)}")
        if len(problems) > 20:
            console.print(f"    ... and {len(problems) - 20} more")
    if not prompts:
        console.print("[red]✗ No valid prompts to process.[/red]")
        raise typer.Exit(code=1)

    try:
        client = DaemonClient(host=host, port=port)
    except DaemonClientError as err:
        handle_cli_error(err)

    start_time = time.time()

    async def _run() -> list[dict[str, Any]]:
        async with httpx.AsyncClient(
            base_url=client.base_url, timeout=client.build_timeout(long=True)
        ) as http:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Processing...", total=len(prompts))
                return await process_prompts(
                    http,
                    model,
                    prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    batch_size=batch_size,
                    on_done=lambda: progress.advance(task),
                )

    results = asyncio.run(_run())
    elapsed = time.time() - start_time
    total_tokens = sum(int(r.get("tokens", 0)) for r in results)
    failed = [r for r in results if "error" in r]

    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    console.print("\n[green]✓ Batch complete[/green]" if not failed else "\n[yellow]Batch finished with errors[/yellow]")
    console.print(f"  Processed: {len(results)} prompts ({len(failed)} failed)")
    console.print(f"  Tokens:    {total_tokens:,}")
    console.print(f"  Time:      {elapsed:.1f}s")
    if elapsed > 0:
        console.print(f"  Speed:     {total_tokens / elapsed:.0f} tok/s")
    console.print(f"  Output:    {output_file}")
    if failed:
        console.print(f"[red]  First error:[/red] {escape(str(failed[0]['error']))}")
    if len(failed) == len(results):
        raise typer.Exit(code=1)
