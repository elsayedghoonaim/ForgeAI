"""
ForgeAI TurboQuant benchmarking harness package.

Provides data schemas, profile definitions, pure evaluation gates, and hardware runner for TurboQuant POC.
"""

from typing import Any

from forgeai.benchmarking.runner import (
    PreflightResult,
    SequentialBenchmarkRunner,
    calculate_percentile,
    default_gpu_memory_used_query,
    load_quality_evidence,
    parse_gpu_memory_used_output,
    parse_kv_cache_capacity_from_log,
    parse_log_anomalies,
    parse_sse_stream_chunk,
    process_streaming_response,
    run_preflight,
)
from forgeai.benchmarking import turboquant as _turboquant
from forgeai.benchmarking.turboquant import (
    PROFILES_CATALOG,
    SCHEMA_VERSION,
    STATUS_FAIL,
    STATUS_INCOMPLETE,
    STATUS_NOT_RUN,
    STATUS_PASS,
    VLLM_CONTRACT_SPEC,
    VLLM_PINNED_VERSION,
    BenchmarkProfileSpec,
    FutureCommandPlan,
    GateOutcome,
    HardwareMetadata,
    ProfileMeasurements,
    ProfileResult,
    TurboQuantBenchmarkArtifact,
    create_benchmark_plan,
    evaluate_artifact as _evaluate_artifact_raw,
    parse_cuda_compute_capability,
)


def evaluate_artifact(
    artifact: TurboQuantBenchmarkArtifact | dict[str, Any] | str,
) -> TurboQuantBenchmarkArtifact:
    """Evaluate an artifact while preserving `not_run` for untouched plan templates.

    A generated plan already contains profile shells, but none have measurements.
    The core evaluator correctly marks its contract evidence incomplete; for an
    unvalidated template this is not an attempted/partial run, so the overall
    artifact status remains `not_run` as documented by the public contract.
    """
    evaluated = _evaluate_artifact_raw(artifact)
    profiles = list(evaluated.profile_results.values())
    untouched = not profiles or all(profile.measurements is None for profile in profiles)
    if (
        not evaluated.hardware_validation_performed
        and untouched
        and evaluated.status == STATUS_INCOMPLETE
    ):
        evaluated.status = STATUS_NOT_RUN
    return evaluated


# Direct imports from forgeai.benchmarking.turboquant should observe the same
# public semantics as imports from this package.
_turboquant.evaluate_artifact = evaluate_artifact


__all__ = [
    "VLLM_PINNED_VERSION",
    "VLLM_CONTRACT_SPEC",
    "SCHEMA_VERSION",
    "STATUS_NOT_RUN",
    "STATUS_INCOMPLETE",
    "STATUS_PASS",
    "STATUS_FAIL",
    "BenchmarkProfileSpec",
    "PROFILES_CATALOG",
    "HardwareMetadata",
    "ProfileMeasurements",
    "ProfileResult",
    "FutureCommandPlan",
    "GateOutcome",
    "TurboQuantBenchmarkArtifact",
    "create_benchmark_plan",
    "evaluate_artifact",
    "parse_cuda_compute_capability",
    "PreflightResult",
    "SequentialBenchmarkRunner",
    "run_preflight",
    "load_quality_evidence",
    "calculate_percentile",
    "parse_kv_cache_capacity_from_log",
    "parse_log_anomalies",
    "default_gpu_memory_used_query",
    "parse_gpu_memory_used_output",
    "parse_sse_stream_chunk",
    "process_streaming_response",
]
