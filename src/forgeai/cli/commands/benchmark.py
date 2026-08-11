"""
forgeai benchmark — TurboQuant benchmark CLI frontend.

Provides plan, evaluate, and execute action modes for TurboQuant benchmark artifacts.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console()
app = typer.Typer(invoke_without_command=True)


@app.callback(invoke_without_command=True)
def benchmark(
    model: str = typer.Argument(
        "Qwen/Qwen3-0.6B",
        help="Target Hugging Face model repository or local directory path.",
    ),
    mode: str = typer.Option(
        "plan",
        "--mode",
        "-m",
        help="Benchmark action mode: plan (default), evaluate, or execute.",
    ),
    input_path: str | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to input benchmark JSON artifact (required for evaluate mode).",
    ),
    output_path: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output JSON path for benchmark plan or evaluated artifact.",
    ),
    output_dir: str = typer.Option(
        "./artifacts/turboquant",
        "--output-dir",
        help="Base output directory for profile log outputs and artifacts.",
    ),
    work_dir: str = typer.Option(
        "/tmp/forgeai_turboquant_bench",
        "--work-dir",
        help="Base working directory for per-profile result/artifact directories in plan mode.",
    ),
    shared_cache: str = typer.Option(
        "~/.forgeai/hf",
        "--shared-cache",
        "--model-cache-dir",
        help="Shared model cache directory passed via --download-dir to all commands.",
    ),
    quality_evidence: str | None = typer.Option(
        None,
        "--quality-evidence",
        help="Path to local quality evidence JSON file (measured perplexity and task accuracies).",
    ),
    dtype: str = typer.Option(
        "bfloat16",
        "--dtype",
        help="Engine / model dtype (default: bfloat16).",
    ),
    weight_quantization: str = typer.Option(
        "none",
        "--weight-quantization",
        help="Model weight quantization (distinct from KV-cache dtype).",
    ),
    base_port: int = typer.Option(
        11435,
        "--base-port",
        help="Starting port number for sequential execution plans (default: 11435).",
    ),
    iterations: int = typer.Option(
        10,
        "--iterations",
        "-n",
        help="Number of benchmark request iterations per profile (default: 10).",
    ),
    warmup: int = typer.Option(
        2,
        "--warmup",
        help="Number of warmup request iterations per profile (default: 2).",
    ),
    max_tokens: int = typer.Option(
        128,
        "--max-tokens",
        help="Maximum generation tokens per request (default: 128).",
    ),
    readiness_timeout: float = typer.Option(
        600.0,
        "--readiness-timeout",
        help="Maximum readiness polling timeout in seconds (default: 600.0).",
    ),
    shutdown_timeout: float = typer.Option(
        30.0,
        "--shutdown-timeout",
        help="Maximum server shutdown timeout in seconds before force kill (default: 30.0).",
    ),
    soak_duration: float = typer.Option(
        1800.0,
        "--soak-duration",
        help="Stability soak duration in seconds (default: 1800.0).",
    ),
    acknowledge_hardware_run: bool = typer.Option(
        False,
        "--acknowledge-hardware-run",
        help="Explicit mandatory opt-in flag required for hardware benchmark execution.",
    ),
) -> None:
    """TurboQuant performance benchmarking and compute evaluation harness."""
    from forgeai.core.telemetry import track_event

    resolved = model
    if resolved.lower().endswith(".gguf") or ".gguf" in resolved.lower():
        console.print(
            f"[red]ERROR:[/red] GGUF model format is unsupported in ForgeAI v2.0+ (model: {resolved!r}). "
            "llama.cpp has been removed in favor of vLLM. "
            "Remediation: Specify a Hugging Face repo ID or local safetensors directory."
        )
        raise typer.Exit(code=1)

    mode_clean = mode.lower().strip()
    valid_modes = ("plan", "evaluate", "execute")
    if mode_clean not in valid_modes:
        console.print(
            f"[red]ERROR:[/red] Unknown benchmark mode '{mode}'. "
            f"Supported modes are: {', '.join(valid_modes)}."
        )
        raise typer.Exit(code=1)

    track_event("command.benchmark", {"model": resolved, "mode": mode_clean})

    if mode_clean in ("plan", "execute"):
        from forgeai.benchmarking.turboquant import PRIMARY_BENCHMARK_PROFILES

        profile_count = len(PRIMARY_BENCHMARK_PROFILES)
        max_generated_port = base_port + profile_count - 1
        if base_port < 1 or max_generated_port > 65535:
            console.print(
                f"[red]ERROR:[/red] --base-port {base_port} is invalid. "
                f"Sequential profile ports ({base_port}..{max_generated_port}) must all fit within 1..65535."
            )
            raise typer.Exit(code=1)

    if mode_clean == "plan":
        from forgeai.benchmarking.turboquant import create_benchmark_plan

        plan = create_benchmark_plan(
            model=resolved,
            base_port=base_port,
            base_work_dir=work_dir,
            model_cache_dir=shared_cache,
            dtype=dtype,
            weight_quantization=weight_quantization,
        )

        json_str = plan.to_json(indent=2)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json_str, encoding="utf-8")
            console.print(f"[green]Plan artifact saved to:[/green] {out_p}")

        console.print("\n[bold cyan]ForgeAI TurboQuant Benchmark Plan[/bold cyan]")
        console.print(f"  Model Target:                   {plan.model}")
        console.print(f"  Contract vLLM Version:          {plan.vllm_version}")
        console.print(f"  Hardware Validation Performed:  {plan.hardware_validation_performed}")
        console.print(f"  Artifact Status:                [bold yellow]{plan.status}[/bold yellow]\n")

        table = Table(title="Sequential Command Plans", show_lines=True)
        table.add_column("Order", justify="center", style="cyan")
        table.add_column("Profile", style="bold")
        table.add_column("Dtype", style="magenta")
        table.add_column("KV Cache Dtype", style="green")
        table.add_column("Port", justify="right")

        for cp in plan.command_plans:
            table.add_row(
                str(cp.sequential_order),
                cp.profile_name,
                cp.dtype,
                cp.kv_cache_dtype,
                str(cp.port),
            )
        console.print(table)
        console.print("\n[dim]Notice: No hardware validation ran. Plan status is 'not_run'.[/dim]\n")
        return

    elif mode_clean == "evaluate":
        if not input_path:
            console.print("[red]ERROR:[/red] Evaluate mode requires '--input' / '-i' JSON artifact path.")
            raise typer.Exit(code=1)

        in_p = Path(input_path)
        if not in_p.exists():
            console.print(f"[red]ERROR:[/red] Input file '{in_p}' does not exist.")
            raise typer.Exit(code=1)

        from forgeai.benchmarking.turboquant import (
            STATUS_PASS,
            TurboQuantBenchmarkArtifact,
            evaluate_artifact,
        )

        try:
            content = in_p.read_text(encoding="utf-8")
            artifact = TurboQuantBenchmarkArtifact.from_json(content)
        except Exception as e:
            console.print(f"[red]ERROR:[/red] Reading JSON artifact from '{in_p}' failed: {e}")
            raise typer.Exit(code=1) from e

        evaluated = evaluate_artifact(artifact)
        json_str = evaluated.to_json(indent=2)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json_str, encoding="utf-8")
            console.print(f"[green]Evaluated artifact saved to:[/green] {out_p}")

        is_pass = evaluated.status == STATUS_PASS
        status_color = "green" if is_pass else "red"

        console.print("\n[bold cyan]ForgeAI TurboQuant Benchmark Evaluation Summary[/bold cyan]")
        console.print(f"  Model Target:                   {evaluated.model}")
        console.print(f"  Contract vLLM Version:          {evaluated.vllm_version}")
        console.print(f"  Hardware Validation Performed:  {evaluated.hardware_validation_performed}")
        console.print(f"  Overall Status:                 [{status_color}]{evaluated.status.upper()}[/{status_color}]\n")

        table = Table(title="Gate Outcomes & Evaluation Reasons", show_lines=True)
        table.add_column("Gate", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for gate_key, outcome in evaluated.gate_outcomes.items():
            st_str = (
                "[green]PASS[/green]"
                if outcome.passed
                else ("[yellow]INCOMPLETE[/yellow]" if outcome.status == "incomplete" else "[red]FAIL[/red]")
            )
            reasons = "; ".join(outcome.reasons) if outcome.reasons else "OK"
            table.add_row(gate_key, st_str, escape(reasons))

        console.print(table)
        console.print()

        if not is_pass:
            raise typer.Exit(code=1)
        return

    elif mode_clean == "execute":
        if not acknowledge_hardware_run:
            console.print(
                "[red]ERROR:[/red] Hardware execution requires explicit opt-in flag '--acknowledge-hardware-run'. "
                "Exiting before preflight."
            )
            raise typer.Exit(code=1)

        # Bounded range checks
        if iterations <= 0:
            console.print("[red]ERROR:[/red] --iterations must be positive.")
            raise typer.Exit(code=1)
        if warmup < 0:
            console.print("[red]ERROR:[/red] --warmup cannot be negative.")
            raise typer.Exit(code=1)
        if max_tokens <= 0:
            console.print("[red]ERROR:[/red] --max-tokens must be positive.")
            raise typer.Exit(code=1)
        if readiness_timeout <= 0:
            console.print("[red]ERROR:[/red] --readiness-timeout must be positive.")
            raise typer.Exit(code=1)
        if shutdown_timeout <= 0:
            console.print("[red]ERROR:[/red] --shutdown-timeout must be positive.")
            raise typer.Exit(code=1)
        if soak_duration < 0:
            console.print("[red]ERROR:[/red] --soak-duration cannot be negative.")
            raise typer.Exit(code=1)

        from forgeai.benchmarking.runner import SequentialBenchmarkRunner
        from forgeai.benchmarking.turboquant import STATUS_PASS, create_benchmark_plan

        plan = create_benchmark_plan(
            model=resolved,
            base_port=base_port,
            base_work_dir=output_dir,
            model_cache_dir=shared_cache,
            dtype=dtype,
            weight_quantization=weight_quantization,
        )

        runner = SequentialBenchmarkRunner(
            plan_artifact=plan,
            output_dir=output_dir,
            artifact_output_path=output_path,
            quality_evidence_path=quality_evidence,
            readiness_timeout_seconds=readiness_timeout,
            shutdown_timeout_seconds=shutdown_timeout,
            soak_duration_seconds=soak_duration,
            num_iterations=iterations,
            num_warmup=warmup,
            max_tokens=max_tokens,
            acknowledge_hardware_run=acknowledge_hardware_run,
        )

        try:
            evaluated = runner.run()
        except Exception as e:
            console.print(f"[red]ERROR:[/red] Hardware execution failed: {escape(str(e))}")
            raise typer.Exit(code=1) from e

        is_pass = evaluated.status == STATUS_PASS
        status_color = "green" if is_pass else "red"

        console.print("\n[bold cyan]ForgeAI TurboQuant Hardware Execution Summary[/bold cyan]")
        console.print(f"  Model Target:    {evaluated.model}")
        console.print(f"  vLLM Version:    {evaluated.vllm_version}")
        console.print(f"  Hardware Device: {evaluated.hardware.device_name}")
        console.print(f"  Overall Status:  [{status_color}]{evaluated.status.upper()}[/{status_color}]\n")

        if not is_pass:
            raise typer.Exit(code=1)
        return
