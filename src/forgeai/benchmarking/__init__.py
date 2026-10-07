"""ForgeAI TurboQuant benchmarking harness package.

Provides data schemas, profile definitions, pure evaluation gates, and hardware runner for TurboQuant POC.

Imports are lazy (PEP 562) so importing a submodule does not load the whole package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forgeai.benchmarking.runner import (
        PreflightResult as PreflightResult,
    )
    from forgeai.benchmarking.runner import (
        SequentialBenchmarkRunner as SequentialBenchmarkRunner,
    )
    from forgeai.benchmarking.runner import (
        calculate_percentile as calculate_percentile,
    )
    from forgeai.benchmarking.runner import (
        default_gpu_memory_used_query as default_gpu_memory_used_query,
    )
    from forgeai.benchmarking.runner import (
        load_quality_evidence as load_quality_evidence,
    )
    from forgeai.benchmarking.runner import (
        parse_gpu_memory_used_output as parse_gpu_memory_used_output,
    )
    from forgeai.benchmarking.runner import (
        parse_kv_cache_capacity_from_log as parse_kv_cache_capacity_from_log,
    )
    from forgeai.benchmarking.runner import (
        parse_log_anomalies as parse_log_anomalies,
    )
    from forgeai.benchmarking.runner import (
        parse_sse_stream_chunk as parse_sse_stream_chunk,
    )
    from forgeai.benchmarking.runner import (
        process_streaming_response as process_streaming_response,
    )
    from forgeai.benchmarking.runner import (
        run_preflight as run_preflight,
    )
    from forgeai.benchmarking.turboquant import (
        PROFILES_CATALOG as PROFILES_CATALOG,
    )
    from forgeai.benchmarking.turboquant import (
        SCHEMA_VERSION as SCHEMA_VERSION,
    )
    from forgeai.benchmarking.turboquant import (
        STATUS_FAIL as STATUS_FAIL,
    )
    from forgeai.benchmarking.turboquant import (
        STATUS_INCOMPLETE as STATUS_INCOMPLETE,
    )
    from forgeai.benchmarking.turboquant import (
        STATUS_NOT_RUN as STATUS_NOT_RUN,
    )
    from forgeai.benchmarking.turboquant import (
        STATUS_PASS as STATUS_PASS,
    )
    from forgeai.benchmarking.turboquant import (
        VLLM_CONTRACT_SPEC as VLLM_CONTRACT_SPEC,
    )
    from forgeai.benchmarking.turboquant import (
        VLLM_PINNED_VERSION as VLLM_PINNED_VERSION,
    )
    from forgeai.benchmarking.turboquant import (
        BenchmarkProfileSpec as BenchmarkProfileSpec,
    )
    from forgeai.benchmarking.turboquant import (
        FutureCommandPlan as FutureCommandPlan,
    )
    from forgeai.benchmarking.turboquant import (
        GateOutcome as GateOutcome,
    )
    from forgeai.benchmarking.turboquant import (
        HardwareMetadata as HardwareMetadata,
    )
    from forgeai.benchmarking.turboquant import (
        ProfileMeasurements as ProfileMeasurements,
    )
    from forgeai.benchmarking.turboquant import (
        ProfileResult as ProfileResult,
    )
    from forgeai.benchmarking.turboquant import (
        TurboQuantBenchmarkArtifact as TurboQuantBenchmarkArtifact,
    )
    from forgeai.benchmarking.turboquant import (
        create_benchmark_plan as create_benchmark_plan,
    )
    from forgeai.benchmarking.turboquant import (
        evaluate_artifact as evaluate_artifact,
    )
    from forgeai.benchmarking.turboquant import (
        parse_cuda_compute_capability as parse_cuda_compute_capability,
    )

_LAZY: dict[str, str] = {
    "PreflightResult": "forgeai.benchmarking.runner",
    "SequentialBenchmarkRunner": "forgeai.benchmarking.runner",
    "calculate_percentile": "forgeai.benchmarking.runner",
    "default_gpu_memory_used_query": "forgeai.benchmarking.runner",
    "load_quality_evidence": "forgeai.benchmarking.runner",
    "parse_gpu_memory_used_output": "forgeai.benchmarking.runner",
    "parse_kv_cache_capacity_from_log": "forgeai.benchmarking.runner",
    "parse_log_anomalies": "forgeai.benchmarking.runner",
    "parse_sse_stream_chunk": "forgeai.benchmarking.runner",
    "process_streaming_response": "forgeai.benchmarking.runner",
    "run_preflight": "forgeai.benchmarking.runner",
    "PROFILES_CATALOG": "forgeai.benchmarking.turboquant",
    "SCHEMA_VERSION": "forgeai.benchmarking.turboquant",
    "STATUS_FAIL": "forgeai.benchmarking.turboquant",
    "STATUS_INCOMPLETE": "forgeai.benchmarking.turboquant",
    "STATUS_NOT_RUN": "forgeai.benchmarking.turboquant",
    "STATUS_PASS": "forgeai.benchmarking.turboquant",
    "VLLM_CONTRACT_SPEC": "forgeai.benchmarking.turboquant",
    "VLLM_PINNED_VERSION": "forgeai.benchmarking.turboquant",
    "BenchmarkProfileSpec": "forgeai.benchmarking.turboquant",
    "FutureCommandPlan": "forgeai.benchmarking.turboquant",
    "GateOutcome": "forgeai.benchmarking.turboquant",
    "HardwareMetadata": "forgeai.benchmarking.turboquant",
    "ProfileMeasurements": "forgeai.benchmarking.turboquant",
    "ProfileResult": "forgeai.benchmarking.turboquant",
    "TurboQuantBenchmarkArtifact": "forgeai.benchmarking.turboquant",
    "create_benchmark_plan": "forgeai.benchmarking.turboquant",
    "evaluate_artifact": "forgeai.benchmarking.turboquant",
    "parse_cuda_compute_capability": "forgeai.benchmarking.turboquant",
}

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


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
