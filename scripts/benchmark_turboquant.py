#!/usr/bin/env python3
"""
Thin plan, evaluate, and explicit opt-in hardware execution entrypoint for TurboQuant POC harness.

Safe by default:
- 'plan' emits plan/template JSON artifacts with status 'not_run';
- 'evaluate' evaluates supplied completed JSON benchmark artifacts;
- 'execute' provides an explicit opt-in hardware execution path requiring --acknowledge-hardware-run.

DOES NOT launch vLLM, query a GPU, download a model, contact a network service,
or run preflight checks on module import or when invoking plan/evaluate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from forgeai.benchmarking.turboquant import (
    STATUS_PASS,
    TurboQuantBenchmarkArtifact,
    create_benchmark_plan,
    evaluate_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ForgeAI TurboQuant POC benchmark harness (plan/evaluate/execute entrypoint)."
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # plan subcommand
    plan_parser = subparsers.add_parser(
        "plan",
        help="Generate a benchmark plan/template JSON artifact (status: not_run).",
    )
    plan_parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Target Hugging Face model repository or local directory path.",
    )
    plan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path for the benchmark plan artifact.",
    )
    plan_parser.add_argument(
        "--base-port",
        type=int,
        default=11435,
        help="Starting port number for sequential execution plans (default: 11435).",
    )
    plan_parser.add_argument(
        "--work-dir",
        type=str,
        default="/tmp/forgeai_turboquant_bench",
        help="Base working directory for per-profile result/artifact directories.",
    )
    plan_parser.add_argument(
        "--model-cache-dir",
        type=str,
        default="~/.forgeai/hf",
        help="Shared model cache directory passed via --download-dir to all commands.",
    )
    plan_parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Engine / model dtype (default: bfloat16 to control baseline comparison).",
    )
    plan_parser.add_argument(
        "--weight-quantization",
        type=str,
        default="none",
        help="Model weight quantization (distinct from KV-cache dtype).",
    )

    # evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a completed benchmark JSON artifact against TurboQuant decision gates.",
    )
    eval_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to completed benchmark JSON artifact file.",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path for the evaluated artifact with gate outcomes.",
    )

    # execute subcommand
    exec_parser = subparsers.add_parser(
        "execute",
        help="Explicit opt-in hardware execution path for TurboQuant benchmark profiles.",
    )
    exec_parser.add_argument(
        "--acknowledge-hardware-run",
        action="store_true",
        default=False,
        help="Explicit mandatory opt-in flag required for hardware benchmark execution.",
    )
    exec_parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Target Hugging Face model repository or local directory path.",
    )
    exec_parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/turboquant",
        help="Base directory for profile log outputs and benchmark artifacts.",
    )
    exec_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional specific output JSON path for the final benchmark artifact.",
    )
    exec_parser.add_argument(
        "--quality-evidence",
        type=str,
        default=None,
        help="Path to local quality evidence JSON file (measured perplexity and task accuracies).",
    )
    exec_parser.add_argument(
        "--model-cache-dir",
        "--shared-cache",
        type=str,
        default="~/.forgeai/hf",
        help="Shared model cache directory passed via --download-dir to all commands.",
    )
    exec_parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Engine / model dtype (default: bfloat16 to control baseline comparison).",
    )
    exec_parser.add_argument(
        "--base-port",
        type=int,
        default=11435,
        help="Starting port number for sequential execution plans (default: 11435).",
    )
    exec_parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of benchmark request iterations per profile (default: 10).",
    )
    exec_parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup request iterations per profile (default: 2).",
    )
    exec_parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum generation tokens per benchmark request (default: 128).",
    )
    exec_parser.add_argument(
        "--readiness-timeout",
        type=float,
        default=600.0,
        help="Maximum readiness polling timeout in seconds (default: 600.0).",
    )
    exec_parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=30.0,
        help="Maximum server shutdown timeout in seconds before force kill (default: 30.0).",
    )
    exec_parser.add_argument(
        "--soak-duration",
        type=float,
        default=1800.0,
        help="Stability soak duration in seconds (default: 1800.0, minimum 1800 required for pass).",
    )

    return parser


def run_plan(args: argparse.Namespace) -> int:
    plan = create_benchmark_plan(
        model=args.model,
        base_port=args.base_port,
        base_work_dir=args.work_dir,
        model_cache_dir=args.model_cache_dir,
        dtype=args.dtype,
        weight_quantization=args.weight_quantization,
    )

    json_str = plan.to_json(indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"[Plan] Saved benchmark plan template artifact to: {out_path}")

    print("=" * 72)
    print("ForgeAI TurboQuant Benchmark Plan Template")
    print("=" * 72)
    print(f"Model Target:                   {plan.model}")
    print(f"Contract vLLM Version:          {plan.vllm_version}")
    print(f"Hardware Validation Performed:  {plan.hardware_validation_performed}")
    print(f"Artifact Status:                {plan.status}")
    print("-" * 72)
    print("Future Sequential Execution Command Plans (argv arrays):")
    for cp in plan.command_plans:
        print(f"  [{cp.sequential_order}] Profile: {cp.profile_name} (dtype: {cp.dtype}, kv_cache_dtype: {cp.kv_cache_dtype})")
        print(f"      Port:         {cp.port}")
        print(f"      Model Cache:  {cp.model_cache_dir}")
        print(f"      WorkDir:      {cp.work_dir}")
        print(f"      Argv:         {cp.argv}")
    print("-" * 72)
    print("Notice: No hardware validation ran. All plans share one model cache path")
    print("(--download-dir) while keeping per-profile artifact work directories distinct.")
    print("=" * 72)

    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file '{in_path}' does not exist.", file=sys.stderr)
        return 1

    try:
        content = in_path.read_text(encoding="utf-8")
        artifact = TurboQuantBenchmarkArtifact.from_json(content)
    except Exception as e:
        print(f"Error reading JSON artifact from '{in_path}': {e}", file=sys.stderr)
        return 1

    evaluated = evaluate_artifact(artifact)
    json_str = evaluated.to_json(indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json_str, encoding="utf-8")
        print(f"[Evaluate] Saved evaluated artifact to: {out_path}")

    print("=" * 72)
    print("ForgeAI TurboQuant Benchmark Evaluation Summary")
    print("=" * 72)
    print(f"Model Target:                   {evaluated.model}")
    print(f"Contract vLLM Version:          {evaluated.vllm_version}")
    print(f"Hardware Validation Performed:  {evaluated.hardware_validation_performed}")
    print(f"Overall Artifact Status:        {evaluated.status.upper()}")
    print("-" * 72)
    print("Gate Outcomes & Evaluation Reasons:")
    for gate_key, outcome in evaluated.gate_outcomes.items():
        pass_str = "PASS" if outcome.passed else ("INCOMPLETE" if outcome.status == "incomplete" else "FAIL")
        print(f"  - Gate [{gate_key}]: {pass_str}")
        if outcome.failed_metrics:
            print(f"    Failed Metrics: {', '.join(outcome.failed_metrics)}")
        for r in outcome.reasons:
            print(f"    Reason: {r}")

    print("=" * 72)

    return 0 if evaluated.status == STATUS_PASS else 1


def run_execute(args: argparse.Namespace) -> int:
    if not args.acknowledge_hardware_run:
        print(
            "Error: Hardware execution requires explicit opt-in flag '--acknowledge-hardware-run'. "
            "Exiting before preflight.",
            file=sys.stderr,
        )
        return 1

    # Validate nonpositive ranges
    if args.base_port <= 0:
        print("Error: --base-port must be positive.", file=sys.stderr)
        return 1
    if args.iterations <= 0:
        print("Error: --iterations must be positive.", file=sys.stderr)
        return 1
    if args.warmup < 0:
        print("Error: --warmup cannot be negative.", file=sys.stderr)
        return 1
    if args.max_tokens <= 0:
        print("Error: --max-tokens must be positive.", file=sys.stderr)
        return 1
    if args.readiness_timeout <= 0:
        print("Error: --readiness-timeout must be positive.", file=sys.stderr)
        return 1
    if args.shutdown_timeout <= 0:
        print("Error: --shutdown-timeout must be positive.", file=sys.stderr)
        return 1
    if args.soak_duration < 0:
        print("Error: --soak-duration cannot be negative.", file=sys.stderr)
        return 1

    plan = create_benchmark_plan(
        model=args.model,
        base_port=args.base_port,
        base_work_dir=args.output_dir,
        model_cache_dir=args.model_cache_dir,
        dtype=args.dtype,
    )

    from forgeai.benchmarking.runner import SequentialBenchmarkRunner

    runner = SequentialBenchmarkRunner(
        plan_artifact=plan,
        output_dir=args.output_dir,
        artifact_output_path=args.output,
        quality_evidence_path=args.quality_evidence,
        readiness_timeout_seconds=args.readiness_timeout,
        shutdown_timeout_seconds=args.shutdown_timeout,
        soak_duration_seconds=args.soak_duration,
        num_iterations=args.iterations,
        num_warmup=args.warmup,
        max_tokens=args.max_tokens,
        acknowledge_hardware_run=args.acknowledge_hardware_run,
    )

    try:
        evaluated = runner.run()
    except Exception as e:
        print(f"Hardware execution failed: {e}", file=sys.stderr)
        return 1

    print("=" * 72)
    print("ForgeAI TurboQuant Hardware Execution Summary")
    print("=" * 72)
    print(f"Model Target:                   {evaluated.model}")
    print(f"Contract vLLM Version:          {evaluated.vllm_version}")
    print(f"Hardware Device:                {evaluated.hardware.device_name}")
    print(f"Overall Artifact Status:        {evaluated.status.upper()}")
    print("=" * 72)

    return 0 if evaluated.status == STATUS_PASS else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "plan":
        return run_plan(args)
    elif args.subcommand == "evaluate":
        return run_evaluate(args)
    elif args.subcommand == "execute":
        return run_execute(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
