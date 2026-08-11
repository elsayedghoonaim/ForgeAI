"""
TurboQuant proof-of-concept benchmark artifact model and pure decision evaluator.

Defines benchmark profiles, result/artifact schema, required measurements, exact decision
gates, and pure go/no-go evaluation rules relative to BF16 baseline.

Pinned to exact vLLM contract: vllm==0.22.1.
Never claims hardware validation ran without verified empirical measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from typing import Any

# Contract Pinning
VLLM_PINNED_VERSION: str = "0.22.1"
VLLM_CONTRACT_SPEC: str = "vllm==0.22.1"
SCHEMA_VERSION: str = "2.0"

# Status Constants
STATUS_NOT_RUN: str = "not_run"
STATUS_INCOMPLETE: str = "incomplete"
STATUS_PASS: str = "pass"
STATUS_FAIL: str = "fail"

VALID_STATUSES: set[str] = {STATUS_NOT_RUN, STATUS_INCOMPLETE, STATUS_PASS, STATUS_FAIL}


def parse_cuda_compute_capability(
    cc: tuple[int, int] | float | str | None,
) -> tuple[int, int] | None:
    """Parse CUDA compute capability into a (major, minor) integer tuple."""
    if cc is None:
        return None
    if isinstance(cc, (tuple, list)) and len(cc) >= 2:
        return (int(cc[0]), int(cc[1]))
    if isinstance(cc, (float, int)):
        val = float(cc)
        major = int(val)
        minor = int(round((val - major) * 10))
        return (major, minor)
    if isinstance(cc, str):
        parts = cc.strip().split(".")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return (int(parts[0]), int(parts[1]))
        elif len(parts) == 1 and parts[0].isdigit():
            return (int(parts[0]), 0)
    return None


@dataclass(frozen=True)
class BenchmarkProfileSpec:
    """Immutable specification for a KV-cache benchmark profile."""

    name: str
    display_name: str
    kv_cache_dtype: str
    expected_capacity_ratio: float
    dtype: str = "bfloat16"
    is_baseline: bool = False
    is_experimental: bool = False
    is_poc_only: bool = False
    is_aggressive: bool = False
    description: str = ""


PROFILES_CATALOG: dict[str, BenchmarkProfileSpec] = {
    "auto": BenchmarkProfileSpec(
        name="auto",
        display_name="BF16 Baseline (auto)",
        kv_cache_dtype="auto",
        expected_capacity_ratio=1.0,
        dtype="bfloat16",
        is_baseline=True,
        is_experimental=False,
        is_poc_only=False,
        is_aggressive=False,
        description="BF16 baseline profile with KV-cache dtype auto. Resolves from a forced bfloat16 engine dtype (--dtype bfloat16). Serves as reference for evaluation.",
    ),
    "fp8": BenchmarkProfileSpec(
        name="fp8",
        display_name="FP8 Baseline",
        kv_cache_dtype="fp8",
        expected_capacity_ratio=2.0,
        dtype="bfloat16",
        is_baseline=False,
        is_experimental=False,
        is_poc_only=False,
        is_aggressive=False,
        description="FP8 baseline profile for comparison (--dtype bfloat16 --kv-cache-dtype fp8). Provides comparison metrics, separate from TurboQuant approval.",
    ),
    "turboquant_4bit_nc": BenchmarkProfileSpec(
        name="turboquant_4bit_nc",
        display_name="TurboQuant 4-Bit NC",
        kv_cache_dtype="turboquant_4bit_nc",
        expected_capacity_ratio=3.8,
        dtype="bfloat16",
        is_baseline=False,
        is_experimental=True,
        is_poc_only=True,
        is_aggressive=False,
        description="Primary experimental TurboQuant 4-bit non-contiguous KV-cache profile (POC-gated, --dtype bfloat16 --kv-cache-dtype turboquant_4bit_nc).",
    ),
    "turboquant_3bit_nc": BenchmarkProfileSpec(
        name="turboquant_3bit_nc",
        display_name="Aggressive TurboQuant 3-Bit NC",
        kv_cache_dtype="turboquant_3bit_nc",
        expected_capacity_ratio=4.9,
        dtype="bfloat16",
        is_baseline=False,
        is_experimental=True,
        is_poc_only=True,
        is_aggressive=True,
        description="Aggressive POC-only TurboQuant 3-bit non-contiguous KV-cache profile (--dtype bfloat16 --kv-cache-dtype turboquant_3bit_nc). Never production-enabled by this code.",
    ),
}

PRIMARY_BENCHMARK_PROFILES: tuple[str, ...] = ("auto", "fp8", "turboquant_4bit_nc", "turboquant_3bit_nc")


@dataclass
class HardwareMetadata:
    """Hardware metadata associated with a benchmark run."""

    device_name: str = "Unvalidated / Synthetic Environment"
    cuda_compute_capability: str | tuple[int, int] | None = None
    total_vram_mb: float = 0.0
    driver_version: str | None = None
    platform: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareMetadata:
        return cls(
            device_name=data.get("device_name", "Unvalidated / Synthetic Environment"),
            cuda_compute_capability=data.get("cuda_compute_capability"),
            total_vram_mb=float(data.get("total_vram_mb", 0.0)),
            driver_version=data.get("driver_version"),
            platform=data.get("platform", "cuda"),
        )


@dataclass
class ProfileMeasurements:
    """Measured quantitative metrics for a single profile execution."""

    idle_process_rss_mb: float | None = None
    idle_gpu_memory_mb: float | None = None
    model_load_duration_seconds: float | None = None
    cold_load_peak_vram_mb: float | None = None
    cold_load_steady_vram_mb: float | None = None
    kv_cache_capacity_tokens: int | None = None
    ttft_p50_ms: float | None = None
    ttft_p95_ms: float | None = None
    decode_tokens_per_sec: float | None = None
    prompt_throughput_tokens_per_sec: float | None = None
    perplexity: float | None = None
    task_accuracy: float | None = None  # Percentage 0..100 or ratio 0..1
    long_context_accuracy: float | None = None  # Percentage 0..100 or ratio 0..1
    crashes_count: int | None = None
    nans_count: int | None = None
    has_memory_leak: bool | None = None
    post_unload_residual_vram_mb: float | None = None
    soak_duration_seconds: float | None = None
    evidence_sources: dict[str, str] = field(default_factory=dict)
    quality_provenance: dict[str, Any] | None = None

    # Derived comparison metrics vs BF16 baseline
    capacity_ratio_vs_bf16: float | None = None
    ttft_p95_ratio_vs_bf16: float | None = None
    decode_throughput_ratio_vs_bf16: float | None = None
    relative_perplexity_degradation_pct: float | None = None
    task_accuracy_drop_pp: float | None = None
    long_context_accuracy_drop_pp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileMeasurements:
        return cls(
            idle_process_rss_mb=data.get("idle_process_rss_mb"),
            idle_gpu_memory_mb=data.get("idle_gpu_memory_mb"),
            model_load_duration_seconds=data.get("model_load_duration_seconds"),
            cold_load_peak_vram_mb=data.get("cold_load_peak_vram_mb"),
            cold_load_steady_vram_mb=data.get("cold_load_steady_vram_mb"),
            kv_cache_capacity_tokens=data.get("kv_cache_capacity_tokens"),
            ttft_p50_ms=data.get("ttft_p50_ms"),
            ttft_p95_ms=data.get("ttft_p95_ms"),
            decode_tokens_per_sec=data.get("decode_tokens_per_sec"),
            prompt_throughput_tokens_per_sec=data.get("prompt_throughput_tokens_per_sec"),
            perplexity=data.get("perplexity"),
            task_accuracy=data.get("task_accuracy"),
            long_context_accuracy=data.get("long_context_accuracy"),
            crashes_count=data.get("crashes_count"),
            nans_count=data.get("nans_count"),
            has_memory_leak=data.get("has_memory_leak"),
            post_unload_residual_vram_mb=data.get("post_unload_residual_vram_mb"),
            soak_duration_seconds=data.get("soak_duration_seconds"),
            evidence_sources=dict(data.get("evidence_sources", {})),
            quality_provenance=data.get("quality_provenance"),
            capacity_ratio_vs_bf16=data.get("capacity_ratio_vs_bf16"),
            ttft_p95_ratio_vs_bf16=data.get("ttft_p95_ratio_vs_bf16"),
            decode_throughput_ratio_vs_bf16=data.get("decode_throughput_ratio_vs_bf16"),
            relative_perplexity_degradation_pct=data.get("relative_perplexity_degradation_pct"),
            task_accuracy_drop_pp=data.get("task_accuracy_drop_pp"),
            long_context_accuracy_drop_pp=data.get("long_context_accuracy_drop_pp"),
        )


@dataclass
class ProfileResult:
    """Results container for a single benchmark profile."""

    profile_name: str
    kv_cache_dtype: str
    dtype: str = "bfloat16"  # Model / engine dtype
    weight_quantization: str = "none"  # Weight quantization is separate from KV dtype!
    status: str = STATUS_NOT_RUN
    expected_capacity_ratio: float = 1.0
    measurements: ProfileMeasurements | None = None
    evaluation_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "kv_cache_dtype": self.kv_cache_dtype,
            "dtype": self.dtype,
            "weight_quantization": self.weight_quantization,
            "status": self.status,
            "expected_capacity_ratio": self.expected_capacity_ratio,
            "measurements": self.measurements.to_dict() if self.measurements else None,
            "evaluation_notes": list(self.evaluation_notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileResult:
        meas_data = data.get("measurements")
        meas = ProfileMeasurements.from_dict(meas_data) if meas_data is not None else None
        return cls(
            profile_name=data["profile_name"],
            kv_cache_dtype=data.get("kv_cache_dtype", data["profile_name"]),
            dtype=data.get("dtype", "bfloat16"),
            weight_quantization=data.get("weight_quantization", "none"),
            status=data.get("status", STATUS_NOT_RUN),
            expected_capacity_ratio=float(data.get("expected_capacity_ratio", 1.0)),
            measurements=meas,
            evaluation_notes=list(data.get("evaluation_notes", [])),
        )


@dataclass
class FutureCommandPlan:
    """Structured execution plan for running benchmark commands sequentially."""

    profile_name: str
    kv_cache_dtype: str
    dtype: str
    model_cache_dir: str
    port: int
    work_dir: str
    argv: list[str]
    sequential_order: int
    execution_mode: str = "sequential"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FutureCommandPlan:
        return cls(
            profile_name=data["profile_name"],
            kv_cache_dtype=data["kv_cache_dtype"],
            dtype=data.get("dtype", "bfloat16"),
            model_cache_dir=data.get("model_cache_dir", "~/.forgeai/hf"),
            port=int(data["port"]),
            work_dir=data["work_dir"],
            argv=list(data["argv"]),
            sequential_order=int(data["sequential_order"]),
            execution_mode=data.get("execution_mode", "sequential"),
            notes=data.get("notes", ""),
        )


@dataclass
class GateOutcome:
    """Outcome for a specific decision gate."""

    gate_name: str
    passed: bool
    status: str  # "pass", "fail", "incomplete", "not_run"
    failed_metrics: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateOutcome:
        return cls(
            gate_name=data["gate_name"],
            passed=bool(data["passed"]),
            status=data.get("status", STATUS_NOT_RUN),
            failed_metrics=list(data.get("failed_metrics", [])),
            reasons=list(data.get("reasons", [])),
            details=dict(data.get("details", {})),
        )


@dataclass
class TurboQuantBenchmarkArtifact:
    """
    Deterministic benchmark artifact for TurboQuant POC harness.

    Includes schema version, timestamp, model, hardware metadata, exact vLLM version,
    profile results, gate outcomes, and hardware_validation_performed flag.
    """

    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = "Qwen/Qwen3-0.6B"
    vllm_version: str = VLLM_CONTRACT_SPEC
    hardware_validation_performed: bool = False
    status: str = STATUS_NOT_RUN
    hardware: HardwareMetadata = field(default_factory=HardwareMetadata)
    profile_results: dict[str, ProfileResult] = field(default_factory=dict)
    command_plans: list[FutureCommandPlan] = field(default_factory=list)
    gate_outcomes: dict[str, GateOutcome] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "timestamp": self.timestamp,
            "model": self.model,
            "vllm_version": self.vllm_version,
            "hardware_validation_performed": self.hardware_validation_performed,
            "status": self.status,
            "hardware": self.hardware.to_dict(),
            "profile_results": {k: v.to_dict() for k, v in self.profile_results.items()},
            "command_plans": [p.to_dict() for p in self.command_plans],
            "gate_outcomes": {k: v.to_dict() for k, v in self.gate_outcomes.items()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TurboQuantBenchmarkArtifact:
        hw = HardwareMetadata.from_dict(data.get("hardware", {}))
        profiles = {
            k: ProfileResult.from_dict(v)
            for k, v in data.get("profile_results", {}).items()
        }
        plans = [FutureCommandPlan.from_dict(p) for p in data.get("command_plans", [])]
        gates = {
            k: GateOutcome.from_dict(v)
            for k, v in data.get("gate_outcomes", {}).items()
        }

        return cls(
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            model=data.get("model", "Qwen/Qwen3-0.6B"),
            vllm_version=data.get("vllm_version", VLLM_CONTRACT_SPEC),
            hardware_validation_performed=bool(data.get("hardware_validation_performed", False)),
            status=data.get("status", STATUS_NOT_RUN),
            hardware=hw,
            profile_results=profiles,
            command_plans=plans,
            gate_outcomes=gates,
        )

    @classmethod
    def from_json(cls, json_str: str) -> TurboQuantBenchmarkArtifact:
        return cls.from_dict(json.loads(json_str))


def create_benchmark_plan(
    model: str = "Qwen/Qwen3-0.6B",
    base_port: int = 11435,
    base_work_dir: str = "/tmp/forgeai_turboquant_bench",
    model_cache_dir: str = "~/.forgeai/hf",
    dtype: str = "bfloat16",
    weight_quantization: str = "none",
) -> TurboQuantBenchmarkArtifact:
    """
    Create a deterministic plan/template artifact with status 'not_run'.

    Generates argv arrays for sequential execution of each profile.
    Uses model_cache_dir (--download-dir) as the shared cache for all profile commands,
    while maintaining separate per-profile work_dir locations for artifacts/results.

    Forces --dtype bfloat16 by default so baseline comparisons isolate KV-cache dtype.
    Does NOT launch vLLM, query GPU, download models, or contact network.
    """
    command_plans: list[FutureCommandPlan] = []
    profile_results: dict[str, ProfileResult] = {}

    for idx, name in enumerate(PRIMARY_BENCHMARK_PROFILES, start=1):
        spec = PROFILES_CATALOG[name]
        port = base_port + idx - 1
        work_dir = f"{base_work_dir}/{name}"

        argv = [
            "vllm",
            "serve",
            model,
            "--port",
            str(port),
            "--dtype",
            dtype,
            "--kv-cache-dtype",
            spec.kv_cache_dtype,
            "--download-dir",
            model_cache_dir,
        ]

        if weight_quantization != "none":
            argv.extend(["--quantization", weight_quantization])

        plan = FutureCommandPlan(
            profile_name=name,
            kv_cache_dtype=spec.kv_cache_dtype,
            dtype=dtype,
            model_cache_dir=model_cache_dir,
            port=port,
            work_dir=work_dir,
            argv=argv,
            sequential_order=idx,
            execution_mode="sequential",
            notes=(
                f"Sequential execution step {idx} for {spec.display_name}. "
                f"Uses shared model cache '{model_cache_dir}' and profile work directory '{work_dir}'."
            ),
        )
        command_plans.append(plan)

        profile_results[name] = ProfileResult(
            profile_name=name,
            kv_cache_dtype=spec.kv_cache_dtype,
            dtype=dtype,
            weight_quantization=weight_quantization,
            status=STATUS_NOT_RUN,
            expected_capacity_ratio=spec.expected_capacity_ratio,
            measurements=None,
            evaluation_notes=[f"Expected catalog capacity ratio: {spec.expected_capacity_ratio}x"],
        )

    artifact = TurboQuantBenchmarkArtifact(
        model=model,
        vllm_version=VLLM_CONTRACT_SPEC,
        hardware_validation_performed=False,
        status=STATUS_NOT_RUN,
        hardware=HardwareMetadata(device_name="Unvalidated Plan / Template"),
        profile_results=profile_results,
        command_plans=command_plans,
        gate_outcomes={},
    )
    return artifact


def _check_missing_fields(
    meas: ProfileMeasurements,
    required_fields: list[str],
) -> tuple[list[str], list[str]]:
    """Check if any required fields in measurements are None, non-finite, or invalid."""
    missing_metrics: list[str] = []
    reasons: list[str] = []
    for field_name in required_fields:
        val = getattr(meas, field_name, None)
        if val is None:
            missing_metrics.append(field_name)
            reasons.append(f"Missing required measurement: {field_name}")
            continue

        if isinstance(val, (int, float)):
            if math.isnan(val) or math.isinf(val):
                missing_metrics.append(field_name)
                reasons.append(f"Non-finite measurement value ({val}) for: {field_name}")
                continue

        if field_name == "perplexity":
            if isinstance(val, (int, float)) and val <= 0.0:
                missing_metrics.append(field_name)
                reasons.append(f"Invalid non-positive perplexity ({val}) for: perplexity")
        elif field_name in ("task_accuracy", "long_context_accuracy"):
            if isinstance(val, (int, float)) and not (0.0 <= val <= 100.0):
                missing_metrics.append(field_name)
                reasons.append(f"Accuracy out of valid range [0, 100] ({val}) for: {field_name}")

    return missing_metrics, reasons


def _compute_accuracy_drop(ref_acc: float, exp_acc: float) -> float:
    """Compute accuracy drop in percentage points for normalized percentage values (0..100)."""
    return ref_acc - exp_acc


def evaluate_artifact(
    artifact: TurboQuantBenchmarkArtifact | dict[str, Any] | str,
) -> TurboQuantBenchmarkArtifact:
    """
    Pure deterministic go/no-go evaluator relative to BF16 baseline.

    GATES CONTRACT IDENTITY FIRST:
    - schema_version == SCHEMA_VERSION ("2.0")
    - vllm_version == VLLM_CONTRACT_SPEC ("vllm==0.22.1")
    - hardware_validation_performed is True
    - hardware platform is CUDA
    - total_vram_mb > 0
    - CUDA compute capability is parseable and >= 7.5
    - Profile kv_cache_dtype matches catalog specs

    Evaluates TQ4 and TQ3 profiles against exact decision gate thresholds:
    - TQ4: capacity >= 3.0x; TTFT p95 <= 1.25x; decode >= 80%; relative PPL degradation <= +3%;
           task and long-context drop <= 1.0 pp; zero crashes/NaNs/memory leak;
           post-unload residual VRAM <= max(100 MiB, 2% total VRAM).
    - TQ3: capacity >= 4.0x; TTFT p95 <= 1.40x; decode >= 70%; relative PPL degradation <= +5%;
           task and long-context drop <= 2.0 pp; same stability/reclamation gate.
           TQ3 is marked aggressive and POC-only (never production-enabled).

    Missing evidence or wrong contract identity must never pass.
    For unvalidated plan templates (hardware_validation_performed=False), absent hardware evidence
    yields contract status 'incomplete' and overall status 'not_run', not 'fail'.
    """
    if isinstance(artifact, str):
        artifact_obj = TurboQuantBenchmarkArtifact.from_json(artifact)
    elif isinstance(artifact, dict):
        artifact_obj = TurboQuantBenchmarkArtifact.from_dict(artifact)
    else:
        artifact_obj = artifact

    # Make deep copy / clean instance for evaluation
    eval_art = TurboQuantBenchmarkArtifact.from_dict(artifact_obj.to_dict())

    gate_outcomes: dict[str, GateOutcome] = {}
    profile_results = eval_art.profile_results
    total_vram_mb = eval_art.hardware.total_vram_mb if eval_art.hardware else 0.0
    max_allowed_unload_vram = max(100.0, 0.02 * total_vram_mb) if total_vram_mb > 0 else 100.0

    # ---------------------------------------------------------
    # CONTRACT IDENTITY GATE (Evaluated first before any pass)
    # ---------------------------------------------------------
    contract_failed_metrics: list[str] = []
    contract_reasons: list[str] = []
    contract_is_incompatible = False

    # 1. Schema version
    if eval_art.schema_version != SCHEMA_VERSION:
        contract_failed_metrics.append("schema_version")
        contract_reasons.append(
            f"Schema version '{eval_art.schema_version}' does not match expected '{SCHEMA_VERSION}'."
        )
        contract_is_incompatible = True

    # 2. vLLM contract spec
    if eval_art.vllm_version != VLLM_CONTRACT_SPEC:
        contract_failed_metrics.append("vllm_version")
        contract_reasons.append(
            f"vLLM version '{eval_art.vllm_version}' does not match pinned contract '{VLLM_CONTRACT_SPEC}'."
        )
        contract_is_incompatible = True

    # 3. Hardware validation performed flag
    if not eval_art.hardware_validation_performed:
        contract_failed_metrics.append("hardware_validation_performed")
        contract_reasons.append(
            "Hardware validation has not been performed (hardware_validation_performed is False)."
        )

    # 4. Hardware platform (CUDA required)
    plat = eval_art.hardware.platform.lower().strip() if eval_art.hardware and eval_art.hardware.platform else ""
    if plat != "cuda":
        contract_failed_metrics.append("hardware_platform")
        contract_reasons.append(
            f"Platform '{eval_art.hardware.platform if eval_art.hardware else None}' is unsupported for TurboQuant. Must be 'cuda'."
        )
        contract_is_incompatible = True

    # 5. Total VRAM > 0
    if total_vram_mb <= 0.0:
        contract_failed_metrics.append("total_vram_mb")
        contract_reasons.append(
            f"Total VRAM ({total_vram_mb} MiB) must be greater than zero."
        )
        # Only mark as hard contract failure if hardware validation was claimed as performed
        if eval_art.hardware_validation_performed:
            contract_is_incompatible = True

    # 6. CUDA compute capability >= 7.5
    raw_cc = eval_art.hardware.cuda_compute_capability if eval_art.hardware else None
    parsed_cc = parse_cuda_compute_capability(raw_cc)
    if parsed_cc is None:
        contract_failed_metrics.append("cuda_compute_capability")
        contract_reasons.append("CUDA compute capability is absent or unparseable.")
        # Only mark as hard contract failure if hardware validation was claimed as performed
        if eval_art.hardware_validation_performed:
            contract_is_incompatible = True
    elif parsed_cc < (7, 5):
        contract_failed_metrics.append("cuda_compute_capability")
        contract_reasons.append(
            f"CUDA compute capability {parsed_cc[0]}.{parsed_cc[1]} is below required minimum 7.5."
        )
        contract_is_incompatible = True

    # 7. Profile kv_cache_dtype catalog matching
    for p_name in PRIMARY_BENCHMARK_PROFILES:
        prof_item = profile_results.get(p_name)
        expected_spec = PROFILES_CATALOG[p_name]
        if not prof_item:
            contract_failed_metrics.append("profile_kv_cache_dtype_mismatch")
            contract_reasons.append(f"Required profile '{p_name}' missing from artifact.")
            contract_is_incompatible = True
        elif prof_item.kv_cache_dtype != expected_spec.kv_cache_dtype:
            contract_failed_metrics.append("profile_kv_cache_dtype_mismatch")
            contract_reasons.append(
                f"Profile '{p_name}' kv_cache_dtype '{prof_item.kv_cache_dtype}' does not match expected '{expected_spec.kv_cache_dtype}'."
            )
            contract_is_incompatible = True

    contract_passed = len(contract_failed_metrics) == 0
    if contract_passed:
        contract_status = STATUS_PASS
    elif contract_is_incompatible:
        contract_status = STATUS_FAIL
    else:
        contract_status = STATUS_INCOMPLETE

    gate_outcomes["contract"] = GateOutcome(
        gate_name="contract_identity",
        passed=contract_passed,
        status=contract_status,
        failed_metrics=contract_failed_metrics,
        reasons=contract_reasons,
        details={
            "schema_version": eval_art.schema_version,
            "vllm_version": eval_art.vllm_version,
            "hardware_validation_performed": eval_art.hardware_validation_performed,
            "platform": eval_art.hardware.platform if eval_art.hardware else None,
            "total_vram_mb": total_vram_mb,
            "cuda_compute_capability": f"{parsed_cc[0]}.{parsed_cc[1]}" if parsed_cc else None,
        },
    )

    # Standard required measurement fields for complete evaluation
    REQUIRED_MEASUREMENT_FIELDS = [
        "idle_process_rss_mb",
        "idle_gpu_memory_mb",
        "model_load_duration_seconds",
        "cold_load_peak_vram_mb",
        "cold_load_steady_vram_mb",
        "kv_cache_capacity_tokens",
        "ttft_p50_ms",
        "ttft_p95_ms",
        "decode_tokens_per_sec",
        "prompt_throughput_tokens_per_sec",
        "perplexity",
        "task_accuracy",
        "long_context_accuracy",
        "crashes_count",
        "nans_count",
        "has_memory_leak",
        "post_unload_residual_vram_mb",
        "soak_duration_seconds",
    ]

    # ---------------------------------------------------------
    # PROFILE MEASUREMENT GATES
    # ---------------------------------------------------------

    # 1. Evaluate BF16 baseline ("auto")
    auto_prof = profile_results.get("auto")
    auto_meas = auto_prof.measurements if auto_prof else None
    auto_valid = False

    if not auto_prof or not auto_meas:
        gate_outcomes["auto"] = GateOutcome(
            gate_name="auto_baseline",
            passed=False,
            status=STATUS_NOT_RUN if not auto_prof else STATUS_INCOMPLETE,
            failed_metrics=["auto_measurements"],
            reasons=["Missing BF16 baseline evidence."],
            details={},
        )
        if auto_prof:
            auto_prof.status = STATUS_INCOMPLETE
    else:
        missing_auto, missing_auto_reasons = _check_missing_fields(auto_meas, REQUIRED_MEASUREMENT_FIELDS)
        if missing_auto:
            gate_outcomes["auto"] = GateOutcome(
                gate_name="auto_baseline",
                passed=False,
                status=STATUS_INCOMPLETE,
                failed_metrics=missing_auto,
                reasons=missing_auto_reasons,
                details={},
            )
            auto_prof.status = STATUS_INCOMPLETE
        else:
            # Check stability & reclamation for BF16 baseline
            auto_failed_metrics: list[str] = []
            auto_reasons: list[str] = []

            if auto_meas.crashes_count != 0 or auto_meas.nans_count != 0 or auto_meas.has_memory_leak:
                auto_failed_metrics.append("stability")
                auto_reasons.append(
                    f"BF16 baseline stability failure: crashes={auto_meas.crashes_count}, "
                    f"nans={auto_meas.nans_count}, leak={auto_meas.has_memory_leak}"
                )

            if auto_meas.soak_duration_seconds is None or auto_meas.soak_duration_seconds < 1800.0:
                auto_failed_metrics.append("soak_duration_seconds")
                auto_reasons.append(
                    f"BF16 baseline stability soak duration ({auto_meas.soak_duration_seconds}s) is below required minimum 1800.0s."
                )

            if auto_meas.post_unload_residual_vram_mb > max_allowed_unload_vram:
                auto_failed_metrics.append("post_unload_residual_vram")
                auto_reasons.append(
                    f"BF16 baseline unload residual {auto_meas.post_unload_residual_vram_mb:.1f} MiB "
                    f"exceeds limit {max_allowed_unload_vram:.1f} MiB"
                )

            if auto_failed_metrics:
                gate_outcomes["auto"] = GateOutcome(
                    gate_name="auto_baseline",
                    passed=False,
                    status=STATUS_FAIL,
                    failed_metrics=auto_failed_metrics,
                    reasons=auto_reasons,
                    details={},
                )
                auto_prof.status = STATUS_FAIL
            else:
                gate_outcomes["auto"] = GateOutcome(
                    gate_name="auto_baseline",
                    passed=True,
                    status=STATUS_PASS,
                    failed_metrics=[],
                    reasons=["BF16 baseline complete and passed stability gate."],
                    details={},
                )
                auto_prof.status = STATUS_PASS
                auto_valid = True

    # Helper function to evaluate experimental TurboQuant profile against auto baseline
    def _evaluate_tq_profile(
        profile_name: str,
        gate_key: str,
        target_cap_ratio: float,
        max_ttft_ratio: float,
        min_decode_ratio: float,
        max_rel_ppl_deg_pct: float,
        max_acc_drop_pp: float,
        is_aggressive_poc_only: bool = False,
    ) -> None:
        prof = profile_results.get(profile_name)
        if not prof:
            gate_outcomes[gate_key] = GateOutcome(
                gate_name=gate_key,
                passed=False,
                status=STATUS_NOT_RUN,
                failed_metrics=[profile_name],
                reasons=[f"Profile '{profile_name}' not present in artifact."],
            )
            return

        meas = prof.measurements
        if not meas:
            gate_outcomes[gate_key] = GateOutcome(
                gate_name=gate_key,
                passed=False,
                status=STATUS_NOT_RUN,
                failed_metrics=[f"{profile_name}_measurements"],
                reasons=[f"Missing measurements for '{profile_name}'."],
            )
            prof.status = STATUS_NOT_RUN
            return

        missing_fields, missing_reasons = _check_missing_fields(meas, REQUIRED_MEASUREMENT_FIELDS)
        if missing_fields:
            gate_outcomes[gate_key] = GateOutcome(
                gate_name=gate_key,
                passed=False,
                status=STATUS_INCOMPLETE,
                failed_metrics=missing_fields,
                reasons=missing_reasons,
            )
            prof.status = STATUS_INCOMPLETE
            return

        if not auto_valid or not auto_meas:
            gate_outcomes[gate_key] = GateOutcome(
                gate_name=gate_key,
                passed=False,
                status=STATUS_INCOMPLETE,
                failed_metrics=["bf16_baseline_reference"],
                reasons=[f"Cannot evaluate '{profile_name}': BF16 baseline reference evidence is missing or invalid."],
            )
            prof.status = STATUS_INCOMPLETE
            return

        # Compute derived metrics vs BF16
        cap_ratio = (
            meas.kv_cache_capacity_tokens / auto_meas.kv_cache_capacity_tokens
            if auto_meas.kv_cache_capacity_tokens and auto_meas.kv_cache_capacity_tokens > 0
            else 0.0
        )
        ttft_ratio = (
            meas.ttft_p95_ms / auto_meas.ttft_p95_ms
            if auto_meas.ttft_p95_ms and auto_meas.ttft_p95_ms > 0
            else 999.0
        )
        decode_ratio = (
            meas.decode_tokens_per_sec / auto_meas.decode_tokens_per_sec
            if auto_meas.decode_tokens_per_sec and auto_meas.decode_tokens_per_sec > 0
            else 0.0
        )
        rel_ppl_deg = (
            ((meas.perplexity - auto_meas.perplexity) / auto_meas.perplexity) * 100.0
            if auto_meas.perplexity and auto_meas.perplexity > 0
            else 999.0
        )
        task_drop_pp = _compute_accuracy_drop(auto_meas.task_accuracy, meas.task_accuracy)
        lc_drop_pp = _compute_accuracy_drop(auto_meas.long_context_accuracy, meas.long_context_accuracy)

        # Store derived metrics on measurements
        meas.capacity_ratio_vs_bf16 = round(cap_ratio, 4)
        meas.ttft_p95_ratio_vs_bf16 = round(ttft_ratio, 4)
        meas.decode_throughput_ratio_vs_bf16 = round(decode_ratio, 4)
        meas.relative_perplexity_degradation_pct = round(rel_ppl_deg, 4)
        meas.task_accuracy_drop_pp = round(task_drop_pp, 4)
        meas.long_context_accuracy_drop_pp = round(lc_drop_pp, 4)

        failed_metrics: list[str] = []
        reasons: list[str] = []

        if is_aggressive_poc_only:
            reasons.append("Aggressive POC-only profile; never production-enabled.")

        # Gate 1: Capacity ratio
        if cap_ratio < target_cap_ratio:
            failed_metrics.append("kv_cache_capacity_ratio")
            reasons.append(
                f"KV cache capacity ratio {cap_ratio:.2f}x is below required threshold {target_cap_ratio:.2f}x"
            )

        # Gate 2: TTFT p95 ratio
        if ttft_ratio > max_ttft_ratio:
            failed_metrics.append("ttft_p95_ratio")
            reasons.append(
                f"TTFT p95 ratio {ttft_ratio:.2f}x exceeds maximum allowed threshold {max_ttft_ratio:.2f}x"
            )

        # Gate 3: Decode speed ratio
        if decode_ratio < min_decode_ratio:
            failed_metrics.append("decode_throughput_ratio")
            reasons.append(
                f"Decode speed ratio {decode_ratio * 100.0:.1f}% is below required threshold {min_decode_ratio * 100.0:.1f}%"
            )

        # Gate 4: Relative perplexity degradation
        if rel_ppl_deg > max_rel_ppl_deg_pct:
            failed_metrics.append("relative_perplexity_degradation")
            reasons.append(
                f"Relative perplexity degradation +{rel_ppl_deg:.2f}% exceeds maximum allowed threshold +{max_rel_ppl_deg_pct:.2f}%"
            )

        # Gate 5: Task accuracy drop
        if task_drop_pp > max_acc_drop_pp:
            failed_metrics.append("task_accuracy_drop")
            reasons.append(
                f"Task accuracy drop {task_drop_pp:.2f} percentage points exceeds maximum allowed threshold {max_acc_drop_pp:.2f} percentage point"
            )

        # Gate 6: Long-context accuracy drop
        if lc_drop_pp > max_acc_drop_pp:
            failed_metrics.append("long_context_accuracy_drop")
            reasons.append(
                f"Long-context accuracy drop {lc_drop_pp:.2f} percentage points exceeds maximum allowed threshold {max_acc_drop_pp:.2f} percentage point"
            )

        # Gate 7: Stability soak (zero crashes, zero NaNs, zero memory leak, soak >= 1800s)
        if meas.crashes_count != 0 or meas.nans_count != 0 or meas.has_memory_leak:
            failed_metrics.append("stability")
            reasons.append(
                f"Stability gate failed: crashes_count={meas.crashes_count}, "
                f"nans_count={meas.nans_count}, has_memory_leak={meas.has_memory_leak}"
            )
        if meas.soak_duration_seconds is None or meas.soak_duration_seconds < 1800.0:
            failed_metrics.append("soak_duration_seconds")
            reasons.append(
                f"Stability soak duration {meas.soak_duration_seconds}s is below required minimum 1800.0s."
            )

        # Gate 8: VRAM reclamation
        if meas.post_unload_residual_vram_mb > max_allowed_unload_vram:
            failed_metrics.append("post_unload_residual_vram")
            reasons.append(
                f"Post-unload residual VRAM {meas.post_unload_residual_vram_mb:.1f} MiB "
                f"exceeds maximum allowed threshold {max_allowed_unload_vram:.1f} MiB"
            )

        passed = len(failed_metrics) == 0
        status = STATUS_PASS if passed else STATUS_FAIL

        prof.status = status
        prof.evaluation_notes = list(reasons)

        gate_outcomes[gate_key] = GateOutcome(
            gate_name=gate_key,
            passed=passed,
            status=status,
            failed_metrics=failed_metrics,
            reasons=reasons,
            details={
                "capacity_ratio_vs_bf16": cap_ratio,
                "ttft_p95_ratio_vs_bf16": ttft_ratio,
                "decode_throughput_ratio_vs_bf16": decode_ratio,
                "relative_perplexity_degradation_pct": rel_ppl_deg,
                "task_accuracy_drop_pp": task_drop_pp,
                "long_context_accuracy_drop_pp": lc_drop_pp,
            },
        )

    # 2. Evaluate TQ4 (Primary experimental profile)
    _evaluate_tq_profile(
        profile_name="turboquant_4bit_nc",
        gate_key="turboquant_4bit_nc",
        target_cap_ratio=3.0,
        max_ttft_ratio=1.25,
        min_decode_ratio=0.80,
        max_rel_ppl_deg_pct=3.0,
        max_acc_drop_pp=1.0,
        is_aggressive_poc_only=False,
    )

    # 3. Evaluate TQ3 (Aggressive POC-only profile)
    _evaluate_tq_profile(
        profile_name="turboquant_3bit_nc",
        gate_key="turboquant_3bit_nc",
        target_cap_ratio=4.0,
        max_ttft_ratio=1.40,
        min_decode_ratio=0.70,
        max_rel_ppl_deg_pct=5.0,
        max_acc_drop_pp=2.0,
        is_aggressive_poc_only=True,
    )

    # 4. Evaluate FP8 (Baseline comparison metrics, does NOT gate TurboQuant approval)
    fp8_prof = profile_results.get("fp8")
    if fp8_prof and fp8_prof.measurements:
        fp8_meas = fp8_prof.measurements
        missing_fp8, missing_fp8_reasons = _check_missing_fields(fp8_meas, REQUIRED_MEASUREMENT_FIELDS)
        if missing_fp8:
            gate_outcomes["fp8"] = GateOutcome(
                gate_name="fp8_baseline",
                passed=False,
                status=STATUS_INCOMPLETE,
                failed_metrics=missing_fp8,
                reasons=missing_fp8_reasons,
            )
            fp8_prof.status = STATUS_INCOMPLETE
        elif auto_valid and auto_meas:
            fp8_cap = fp8_meas.kv_cache_capacity_tokens / auto_meas.kv_cache_capacity_tokens
            fp8_ttft = fp8_meas.ttft_p95_ms / auto_meas.ttft_p95_ms
            fp8_decode = fp8_meas.decode_tokens_per_sec / auto_meas.decode_tokens_per_sec
            fp8_ppl = ((fp8_meas.perplexity - auto_meas.perplexity) / auto_meas.perplexity) * 100.0

            fp8_meas.capacity_ratio_vs_bf16 = round(fp8_cap, 4)
            fp8_meas.ttft_p95_ratio_vs_bf16 = round(fp8_ttft, 4)
            fp8_meas.decode_throughput_ratio_vs_bf16 = round(fp8_decode, 4)
            fp8_meas.relative_perplexity_degradation_pct = round(fp8_ppl, 4)

            fp8_prof.status = STATUS_PASS
            gate_outcomes["fp8"] = GateOutcome(
                gate_name="fp8_baseline",
                passed=True,
                status=STATUS_PASS,
                failed_metrics=[],
                reasons=["FP8 baseline comparison metrics computed. Separate from TurboQuant approval."],
                details={
                    "capacity_ratio_vs_bf16": fp8_cap,
                    "ttft_p95_ratio_vs_bf16": fp8_ttft,
                    "decode_throughput_ratio_vs_bf16": fp8_decode,
                    "relative_perplexity_degradation_pct": fp8_ppl,
                },
            )

    eval_art.gate_outcomes = gate_outcomes

    # Determine overall artifact status
    all_statuses = [p.status for p in profile_results.values()]

    if not contract_passed:
        if contract_status == STATUS_FAIL:
            eval_art.status = STATUS_FAIL
        elif all(s == STATUS_NOT_RUN for s in all_statuses):
            eval_art.status = STATUS_NOT_RUN
        else:
            eval_art.status = STATUS_INCOMPLETE
    else:
        # Contract passed, check profile gates
        tq4_outcome = gate_outcomes.get("turboquant_4bit_nc")
        auto_outcome = gate_outcomes.get("auto")

        if auto_outcome and auto_outcome.passed and tq4_outcome and tq4_outcome.passed:
            eval_art.status = STATUS_PASS
        elif any(s == STATUS_FAIL for s in all_statuses) or (tq4_outcome and not tq4_outcome.passed):
            eval_art.status = STATUS_FAIL
        else:
            eval_art.status = STATUS_INCOMPLETE

    return eval_art
