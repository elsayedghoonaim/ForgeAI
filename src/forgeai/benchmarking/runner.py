"""
Explicit sequential TurboQuant hardware POC runner for ForgeAI.

Provides implementation-ready, explicit opt-in hardware execution path:
- Preflight safety checks (Linux/WSL2, vLLM==0.22.1, CUDA >= 7.5, VRAM > 0, executables);
- Dedicated GPU used-memory sampling via nvidia-smi --query-gpu=memory.used returning float | None;
- Sequential process lifecycle management (at most one vllm serve child, bounded readiness polling);
- Guaranteed teardown safety in finally blocks (graceful terminate, bounded wait, kill exact child);
- Valid streaming token measurements with stream_options include_usage and text-delta TTFT timing;
- Decode throughput requires completion_tokens > 1;
- Stability soak under active load recording actual monotonic elapsed duration, breaking on failed load;
- Observable error reporting and quality evidence provenance persistence with explicit accuracy_scale;
- Atomic artifact persistence after every profile run.

No subprocess/network/GPU side effects occur on import or during unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

from forgeai.benchmarking.turboquant import (
    PROFILES_CATALOG,
    STATUS_FAIL,
    STATUS_INCOMPLETE,
    STATUS_NOT_RUN,
    STATUS_PASS,
    VLLM_CONTRACT_SPEC,
    VLLM_PINNED_VERSION,
    HardwareMetadata,
    ProfileMeasurements,
    ProfileResult,
    TurboQuantBenchmarkArtifact,
    evaluate_artifact,
    parse_cuda_compute_capability,
)


@dataclass
class PreflightResult:
    """Outcome of execution preflight verification."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    hardware: HardwareMetadata = field(default_factory=HardwareMetadata)
    vllm_version: str = VLLM_CONTRACT_SPEC


def default_wsl_check() -> bool:
    """Check if environment is Linux running under WSL2."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        proc_version = Path("/proc/version")
        if proc_version.exists():
            text = proc_version.read_text(encoding="utf-8").lower()
            return "microsoft" in text or "wsl" in text
    except Exception:
        pass
    return False


def default_gpu_query() -> HardwareMetadata:
    """Query NVIDIA CUDA GPU metadata via nvidia-smi."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    line = res.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        raise ValueError(f"Unexpected nvidia-smi output line: '{line}'")

    dev_name = parts[0]
    drv_ver = parts[1]
    total_vram_mb = float(parts[2])
    cc_raw = parts[3]

    return HardwareMetadata(
        device_name=dev_name,
        cuda_compute_capability=cc_raw,
        total_vram_mb=total_vram_mb,
        driver_version=drv_ver,
        platform="cuda",
    )


def default_gpu_memory_used_query() -> float | None:
    """Query current NVIDIA CUDA GPU used memory in MiB via nvidia-smi. Returns None on failure."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )
        return parse_gpu_memory_used_output(res.stdout)
    except Exception:
        return None


def parse_gpu_memory_used_output(output_text: str) -> float | None:
    """Parse GPU memory used (in MiB) from nvidia-smi output text. Returns None if unparseable."""
    try:
        lines = output_text.strip().splitlines()
        if not lines:
            return None
        val_str = lines[0].strip().split(",")[0].strip()
        val = float(val_str)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    except Exception:
        return None


def run_preflight(
    system_platform: str | None = None,
    wsl_check_fn: Callable[[], bool] | None = None,
    vllm_version_fn: Callable[[], str] | None = None,
    which_fn: Callable[[str], str | None] | None = None,
    gpu_query_fn: Callable[[], HardwareMetadata] | None = None,
) -> PreflightResult:
    """
    Perform preflight safety verification before hardware benchmark execution.

    Verifies:
    1. Environment is Linux or WSL2 (native Windows and CPU execution are rejected);
    2. Installed vLLM version is exactly "0.22.1";
    3. Executables 'vllm' and 'nvidia-smi' exist in PATH;
    4. NVIDIA CUDA GPU metadata exists, VRAM > 0, compute capability is parseable and >= 7.5.
    """
    plat = system_platform if system_platform is not None else sys.platform
    check_wsl = wsl_check_fn if wsl_check_fn is not None else default_wsl_check
    get_vllm_ver = (
        vllm_version_fn
        if vllm_version_fn is not None
        else lambda: importlib.metadata.version("vllm")
    )
    check_which = which_fn if which_fn is not None else shutil.which
    get_gpu = gpu_query_fn if gpu_query_fn is not None else default_gpu_query

    errors: list[str] = []

    # 1. Environment check (Linux / WSL2 required)
    is_linux = plat.startswith("linux")
    is_wsl = check_wsl()
    if not (is_linux or is_wsl):
        errors.append(
            f"Unsupported execution environment '{plat}'. Hardware benchmark requires Linux or WSL2."
        )

    # 2. vLLM version check (strictly 0.22.1)
    installed_ver = ""
    try:
        installed_ver = get_vllm_ver()
        if installed_ver != VLLM_PINNED_VERSION:
            errors.append(
                f"Installed vLLM version '{installed_ver}' does not match required contract version '{VLLM_PINNED_VERSION}'."
            )
    except Exception as e:
        errors.append(f"Failed to detect installed vLLM version: {e}")

    # 3. Executable existence check
    if not check_which("vllm"):
        errors.append("Executable 'vllm' not found in PATH.")
    if not check_which("nvidia-smi"):
        errors.append("Executable 'nvidia-smi' not found in PATH.")

    # 4. GPU metadata check
    hw = HardwareMetadata()
    try:
        hw = get_gpu()
        if hw.total_vram_mb <= 0.0:
            errors.append(f"Total VRAM ({hw.total_vram_mb} MiB) must be greater than zero.")

        parsed_cc = parse_cuda_compute_capability(hw.cuda_compute_capability)
        if parsed_cc is None:
            errors.append(f"CUDA compute capability '{hw.cuda_compute_capability}' is unparseable.")
        elif parsed_cc < (7, 5):
            errors.append(
                f"CUDA compute capability {parsed_cc[0]}.{parsed_cc[1]} is below required minimum 7.5."
            )
    except Exception as e:
        errors.append(f"Failed to query NVIDIA GPU metadata: {e}")

    passed = len(errors) == 0
    return PreflightResult(
        passed=passed,
        errors=errors,
        hardware=hw,
        vllm_version=VLLM_CONTRACT_SPEC if installed_ver == VLLM_PINNED_VERSION else f"vllm=={installed_ver}",
    )


def load_quality_evidence(path: str | Path) -> dict[str, dict[str, Any]]:
    """
    Load local quality evidence JSON file.

    Expects JSON containing per-profile quantitative quality measurements:
    - perplexity (float, > 0)
    - task_accuracy (float)
    - long_context_accuracy (float)
    Plus dataset/evaluator provenance metadata with explicit accuracy_scale ('percent' or 'fraction').

    Normalizes fraction accuracies to percentage points [0, 100].
    Rejects malformed JSON, non-finite values, mixed scales, or missing/invalid provenance.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Quality evidence file '{p}' does not exist.")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Malformed JSON in quality evidence file '{p}': {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Quality evidence file root must be a JSON object.")

    top_provenance = data.get("provenance")
    top_scale = top_provenance.get("accuracy_scale") if isinstance(top_provenance, dict) else None

    profiles_data = data.get("profiles", data)
    if not isinstance(profiles_data, dict):
        raise ValueError("Quality evidence profiles must be a dictionary.")

    result: dict[str, dict[str, Any]] = {}
    required_metrics = ["perplexity", "task_accuracy", "long_context_accuracy"]
    file_scale: str | None = str(top_scale).lower().strip() if top_scale else None

    for profile_name, prof_info in profiles_data.items():
        if profile_name == "provenance":
            continue
        if not isinstance(prof_info, dict):
            continue

        prof_prov = prof_info.get("provenance", {})
        prof_scale = (
            prof_prov.get("accuracy_scale")
            if isinstance(prof_prov, dict)
            else prof_info.get("accuracy_scale")
        )
        effective_scale = prof_scale or file_scale

        if not effective_scale or str(effective_scale).lower().strip() not in ("percent", "fraction"):
            raise ValueError(
                f"Quality evidence provenance for profile '{profile_name}' must declare accuracy_scale as 'percent' or 'fraction'."
            )

        scale_clean = str(effective_scale).lower().strip()
        if file_scale is None:
            file_scale = scale_clean
        elif file_scale != scale_clean:
            raise ValueError(
                f"Inconsistent accuracy_scale in quality evidence: expected '{file_scale}', found '{scale_clean}' in profile '{profile_name}'."
            )

        has_top_prov = (
            isinstance(top_provenance, dict)
            and bool(top_provenance.get("dataset"))
            and bool(top_provenance.get("evaluator"))
        )
        prof_has_prov = (
            (isinstance(prof_prov, dict) and bool(prof_prov.get("dataset")) and bool(prof_prov.get("evaluator")))
            or (bool(prof_info.get("dataset")) and bool(prof_info.get("evaluator")))
        )
        if not (has_top_prov or prof_has_prov):
            raise ValueError(
                f"Quality evidence for profile '{profile_name}' is missing required dataset/evaluator provenance."
            )

        dataset_str = (
            top_provenance.get("dataset")
            if has_top_prov
            else (prof_prov.get("dataset") if isinstance(prof_prov, dict) else prof_info.get("dataset"))
        )
        evaluator_str = (
            top_provenance.get("evaluator")
            if has_top_prov
            else (prof_prov.get("evaluator") if isinstance(prof_prov, dict) else prof_info.get("evaluator"))
        )

        metric_values: dict[str, Any] = {}
        for m in required_metrics:
            if m not in prof_info or prof_info[m] is None:
                raise ValueError(
                    f"Quality evidence for profile '{profile_name}' is missing required metric '{m}'."
                )
            val = float(prof_info[m])
            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"Quality evidence metric '{m}' for profile '{profile_name}' must be a finite number."
                )

            if m == "perplexity":
                if val <= 0.0:
                    raise ValueError(
                        f"Quality evidence perplexity for profile '{profile_name}' must be positive (> 0), got {val}."
                    )
                metric_values[m] = val
            elif m in ("task_accuracy", "long_context_accuracy"):
                if scale_clean == "fraction":
                    if not (0.0 <= val <= 1.0):
                        raise ValueError(
                            f"Quality evidence accuracy '{m}' for profile '{profile_name}' is out of bounds for scale 'fraction' [0, 1]: {val}."
                        )
                    norm_val = val * 100.0
                else:  # percent
                    if not (0.0 <= val <= 100.0):
                        raise ValueError(
                            f"Quality evidence accuracy '{m}' for profile '{profile_name}' is out of bounds for scale 'percent' [0, 100]: {val}."
                        )
                    norm_val = val
                metric_values[m] = norm_val

        metric_values["provenance"] = {
            "dataset": dataset_str,
            "evaluator": evaluator_str,
            "accuracy_scale": scale_clean,
            "normalized_accuracy_scale": "percent",
        }
        result[profile_name] = metric_values

    return result


def calculate_percentile(data: list[float], percentile: float) -> float:
    """
    Calculate deterministic percentile from a list of float values using linear interpolation.

    Formula:
    1. Sort elements in non-decreasing order: x_0, x_1, ..., x_{N-1}
    2. Rank k = (N - 1) * (percentile / 100.0)
    3. Split k into integer part f = floor(k) and fractional part d = k - f
    4. Result = x_f + d * (x_{f+1} - x_f)
    """
    if not data:
        return 0.0
    if len(data) == 1:
        return float(data[0])

    s_data = sorted(data)
    n = len(s_data)
    k = (n - 1) * (percentile / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(s_data[f])
    d = k - f
    return float(s_data[f] + d * (s_data[c] - s_data[f]))


def parse_kv_cache_capacity_from_log(log_text: str) -> int | None:
    """
    Parse KV-cache token capacity from vLLM startup log text.

    Looks for lines matching standard vLLM startup pattern:
    'GPU KV cache size: 123,456 tokens' or 'GPU KV cache size: 123456 tokens'.
    Returns parsed integer token count, or None if missing.
    Never infers capacity from catalog ratios.
    """
    match = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log_text, re.IGNORECASE)
    if match:
        raw_str = match.group(1).replace(",", "")
        try:
            return int(raw_str)
        except ValueError:
            return None
    return None


def parse_log_anomalies(log_text: str) -> tuple[int, int]:
    """
    Parse server log text for NaN occurrences and crash/error markers.

    Returns tuple (nans_count, crashes_count).
    """
    nan_matches = re.findall(r"\b(nan|NaN|NaNs)\b", log_text)
    crash_matches = re.findall(
        r"(Traceback \(most recent call last\)|CUDA error|Segmentation fault|EngineException|RuntimeError: CUDA)",
        log_text,
    )
    return len(nan_matches), len(crash_matches)


def parse_sse_stream_chunk(line_str: str) -> dict[str, Any] | None:
    """Parse a single SSE data line into a dict, returning None for non-JSON / done / non-data lines."""
    line_str = line_str.strip()
    if not line_str.startswith("data:"):
        return None
    data_content = line_str[5:].strip()
    if not data_content or data_content == "[DONE]":
        return None
    try:
        obj = json.loads(data_content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def process_streaming_response(
    chunks: list[tuple[float, str]],
    t0: float,
) -> tuple[float | None, float | None, int | None, int | None]:
    """
    Process streaming SSE chunks from a single completion request.

    TTFT is recorded on the first chunk with non-empty text/content.
    Token counts come from usage object. Decode rate requires completion_tokens > 1.

    Returns: (ttft_ms, decode_time_sec, prompt_tokens, completion_tokens)
    """
    t_first: float | None = None
    t_end: float = t0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    for ts, line in chunks:
        t_end = ts
        data_obj = parse_sse_stream_chunk(line)
        if not data_obj:
            continue

        # Check choices for non-empty text content
        choices = data_obj.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                text_content = choice.get("text")
                if text_content is None and isinstance(choice.get("delta"), dict):
                    text_content = choice.get("delta", {}).get("content")

                if text_content is not None and len(str(text_content)) > 0:
                    if t_first is None:
                        t_first = ts

        # Check usage object
        usage = data_obj.get("usage")
        if isinstance(usage, dict):
            p_tok = usage.get("prompt_tokens")
            c_tok = usage.get("completion_tokens")
            if isinstance(p_tok, int) and p_tok > 0:
                prompt_tokens = p_tok
            if isinstance(c_tok, int) and c_tok > 0:
                completion_tokens = c_tok

    if t_first is None or prompt_tokens is None or completion_tokens is None or completion_tokens <= 1:
        ttft_ms = (t_first - t0) * 1000.0 if t_first is not None else None
        return ttft_ms, None, prompt_tokens, completion_tokens

    ttft_ms = (t_first - t0) * 1000.0
    decode_time_sec = max(0.000001, t_end - t_first)
    return ttft_ms, decode_time_sec, prompt_tokens, completion_tokens


def get_harness_idle_rss_mb() -> float | None:
    """Read current process idle RSS in MiB from /proc/self/statm on Linux."""
    try:
        statm_path = Path("/proc/self/statm")
        if statm_path.exists():
            parts = statm_path.read_text(encoding="utf-8").strip().split()
            if len(parts) >= 2:
                pages = int(parts[1])
                page_size_bytes = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
                rss_bytes = pages * page_size_bytes
                return rss_bytes / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


class SequentialBenchmarkRunner:
    """
    Implementation-ready, explicit opt-in hardware execution coordinator.

    Runs profile command plans sequentially, collects empirical measurements,
    manages server lifecycles safely with guaranteed cleanup in finally blocks,
    and writes atomic benchmark artifacts.
    """

    def __init__(
        self,
        plan_artifact: TurboQuantBenchmarkArtifact,
        output_dir: str | Path = "./artifacts/turboquant",
        artifact_output_path: str | Path | None = None,
        quality_evidence_path: str | Path | None = None,
        readiness_timeout_seconds: float = 600.0,
        shutdown_timeout_seconds: float = 30.0,
        soak_duration_seconds: float = 1800.0,
        num_iterations: int = 10,
        num_warmup: int = 2,
        max_tokens: int = 128,
        acknowledge_hardware_run: bool = False,
        preflight_fn: Callable[[], PreflightResult] | None = None,
        process_factory: Callable[..., Any] | None = None,
        http_client_factory: Callable[[int], Any] | None = None,
        gpu_sampler_fn: Callable[[], float | None] | None = None,
        rss_fn: Callable[[], float | None] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.plan_artifact = plan_artifact
        self.output_dir = Path(output_dir)
        self.artifact_output_path = (
            Path(artifact_output_path)
            if artifact_output_path
            else self.output_dir / "turboquant_benchmark_artifact.json"
        )
        self.quality_evidence_path = Path(quality_evidence_path) if quality_evidence_path else None
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds
        self.soak_duration_seconds = soak_duration_seconds
        self.num_iterations = num_iterations
        self.num_warmup = num_warmup
        self.max_tokens = max_tokens
        self.acknowledge_hardware_run = acknowledge_hardware_run

        self.preflight_fn = preflight_fn if preflight_fn is not None else run_preflight
        self.process_factory = process_factory if process_factory is not None else subprocess.Popen
        self.http_client_factory = http_client_factory
        self.gpu_sampler_fn = (
            gpu_sampler_fn
            if gpu_sampler_fn is not None
            else default_gpu_memory_used_query
        )
        self.rss_fn = rss_fn if rss_fn is not None else get_harness_idle_rss_mb
        self.sleep_fn = sleep_fn if sleep_fn is not None else time.sleep
        self.time_fn = time_fn if time_fn is not None else time.monotonic

    def run(self) -> TurboQuantBenchmarkArtifact:
        """Execute the hardware benchmark sequentially across all profile command plans."""
        # 1. Execution safety check: acknowledge flag is MANDATORY
        if not self.acknowledge_hardware_run:
            raise RuntimeError(
                "Hardware benchmark execution requires explicit opt-in acknowledgment "
                "(--acknowledge-hardware-run). Exiting before preflight."
            )

        # 2. Run preflight verification
        preflight_res = self.preflight_fn()
        if not preflight_res.passed:
            err_msg = "; ".join(preflight_res.errors)
            raise RuntimeError(f"Preflight safety check failed: {err_msg}")

        # 3. Create output directory and load local quality evidence if provided
        self.output_dir.mkdir(parents=True, exist_ok=True)

        quality_data: dict[str, dict[str, Any]] = {}
        if self.quality_evidence_path:
            quality_data = load_quality_evidence(self.quality_evidence_path)

        current_artifact = TurboQuantBenchmarkArtifact.from_dict(self.plan_artifact.to_dict())
        current_artifact.hardware_validation_performed = True
        current_artifact.hardware = preflight_res.hardware
        current_artifact.vllm_version = preflight_res.vllm_version

        # Base idle baseline before runs
        harness_rss = self.rss_fn()
        baseline_idle_vram = self._sample_gpu_vram()

        # 4. Sequentially execute profiles
        for plan in current_artifact.command_plans:
            profile_name = plan.profile_name
            profile_dir = self.output_dir / profile_name
            profile_dir.mkdir(parents=True, exist_ok=True)

            stdout_path = profile_dir / "vllm_stdout.log"
            stderr_path = profile_dir / "vllm_stderr.log"

            prof_meas, eval_notes = self._execute_single_profile(
                plan=plan,
                profile_dir=profile_dir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                harness_rss=harness_rss,
                baseline_idle_vram=baseline_idle_vram,
                total_vram_mb=current_artifact.hardware.total_vram_mb,
                quality_data=quality_data.get(profile_name),
            )

            # Update profile result
            prof_res = current_artifact.profile_results.get(profile_name)
            if not prof_res:
                spec = PROFILES_CATALOG[profile_name]
                prof_res = ProfileResult(
                    profile_name=profile_name,
                    kv_cache_dtype=spec.kv_cache_dtype,
                    dtype=plan.dtype,
                )
                current_artifact.profile_results[profile_name] = prof_res

            prof_res.measurements = prof_meas
            prof_res.evaluation_notes.extend(eval_notes)
            prof_res.status = STATUS_INCOMPLETE  # Will be updated by evaluate_artifact

            # Atomically save artifact progress after every profile
            current_artifact = evaluate_artifact(current_artifact)
            self._save_artifact_atomically(current_artifact)

        # Final evaluation and persistence
        final_evaluated = evaluate_artifact(current_artifact)
        self._save_artifact_atomically(final_evaluated)
        return final_evaluated

    def _sample_gpu_vram(self) -> float | None:
        try:
            val = self.gpu_sampler_fn()
            if val is None or math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            return None

    def _execute_single_profile(
        self,
        plan: Any,
        profile_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
        harness_rss: float | None,
        baseline_idle_vram: float | None,
        total_vram_mb: float,
        quality_data: dict[str, Any] | None,
    ) -> tuple[ProfileMeasurements, list[str]]:
        proc = None
        idle_gpu_mem = self._sample_gpu_vram()
        start_mono = self.time_fn()
        cold_peak_vram = idle_gpu_mem
        readiness_achieved = False
        model_load_duration: float | None = None
        crashes_count = 0
        eval_notes: list[str] = []

        if idle_gpu_mem is None:
            eval_notes.append("GPU memory sampler query failed for idle memory; GPU VRAM measurements are incomplete.")

        stdout_file = stdout_path.open("w", encoding="utf-8")
        stderr_file = stderr_path.open("w", encoding="utf-8")

        try:
            # Subprocess execution using argv list with shell=False
            proc = self.process_factory(
                plan.argv,
                stdout=stdout_file,
                stderr=stderr_file,
                shell=False,
            )

            # Readiness polling loop with bounded timeout
            poll_interval = 0.5
            while (self.time_fn() - start_mono) < self.readiness_timeout_seconds:
                # Check for premature child exit
                exit_code = proc.poll() if hasattr(proc, "poll") else None
                if exit_code is not None:
                    crashes_count += 1
                    eval_notes.append(f"Child process exited prematurely with exit code {exit_code} during readiness polling.")
                    break

                current_vram = self._sample_gpu_vram()
                if current_vram is not None:
                    if cold_peak_vram is None or current_vram > cold_peak_vram:
                        cold_peak_vram = current_vram

                # Check HTTP readiness
                if self._check_http_readiness(plan.port):
                    readiness_achieved = True
                    model_load_duration = self.time_fn() - start_mono
                    break

                self.sleep_fn(poll_interval)

            if not readiness_achieved and crashes_count == 0:
                eval_notes.append(f"Readiness polling timed out after {self.readiness_timeout_seconds} seconds.")

            steady_vram = self._sample_gpu_vram() if readiness_achieved else None
            if readiness_achieved and steady_vram is None:
                eval_notes.append("GPU memory sampler query failed for steady loaded VRAM.")

            # Collect stream measurements if ready
            ttft_p50 = None
            ttft_p95 = None
            decode_tps = None
            prompt_tps = None

            if readiness_achieved:
                ttft_p50, ttft_p95, decode_tps, prompt_tps = self._run_streaming_benchmark(plan.port)
                if ttft_p95 is None or decode_tps is None:
                    eval_notes.append("Streaming HTTP benchmark failed to collect valid metrics or completion_tokens <= 1.")

                # Stability soak under active generation load
                actual_soak_seconds = 0.0
                if self.soak_duration_seconds > 0:
                    soak_start = self.time_fn()
                    while (self.time_fn() - soak_start) < self.soak_duration_seconds:
                        exit_code = proc.poll() if hasattr(proc, "poll") else None
                        if exit_code is not None:
                            crashes_count += 1
                            eval_notes.append(f"Child process exited prematurely with exit code {exit_code} during stability soak.")
                            break

                        soak_ok = self._send_soak_load_request(plan.port)
                        if not soak_ok:
                            exit_code = proc.poll() if hasattr(proc, "poll") else None
                            if exit_code is not None:
                                crashes_count += 1
                                eval_notes.append(f"Child process exited with code {exit_code} during soak load request.")
                            else:
                                eval_notes.append("Soak load request failed while server process was alive; stopping soak accounting immediately.")
                            break

                        self.sleep_fn(min(1.0, self.soak_duration_seconds))

                    actual_soak_seconds = self.time_fn() - soak_start
                soak_recorded_seconds = round(actual_soak_seconds, 2)
            else:
                soak_recorded_seconds = 0.0

        finally:
            if stdout_file:
                stdout_file.close()
            if stderr_file:
                stderr_file.close()

            if proc is not None:
                self._safely_teardown_process(proc)

        # Post-unload residual VRAM
        self.sleep_fn(1.0)
        post_unload_vram = self._sample_gpu_vram()
        if post_unload_vram is not None and baseline_idle_vram is not None:
            post_unload_residual = max(0.0, post_unload_vram - baseline_idle_vram)
        else:
            post_unload_residual = None
            eval_notes.append("GPU memory sampler query failed for post-unload residual VRAM.")

        # Leak threshold uses total_vram_mb matching evaluator
        max_allowed_unload_vram = max(100.0, 0.02 * total_vram_mb) if total_vram_mb > 0 else 100.0
        if post_unload_residual is not None:
            has_memory_leak = post_unload_residual > max_allowed_unload_vram
        else:
            has_memory_leak = None

        # Read and parse logs
        combined_logs = ""
        if stdout_path.exists():
            combined_logs += stdout_path.read_text(encoding="utf-8", errors="replace") + "\n"
        if stderr_path.exists():
            combined_logs += stderr_path.read_text(encoding="utf-8", errors="replace")

        kv_capacity = parse_kv_cache_capacity_from_log(combined_logs)
        nans_count, log_crashes = parse_log_anomalies(combined_logs)
        total_crashes = crashes_count + log_crashes

        # Build evidence sources dictionary
        sources: dict[str, str] = {
            "idle_process_rss_mb": "measured" if harness_rss is not None else "missing",
            "idle_gpu_memory_mb": "measured" if idle_gpu_mem is not None else "missing",
            "model_load_duration_seconds": "measured" if model_load_duration is not None else "missing",
            "cold_load_peak_vram_mb": "measured" if (readiness_achieved and cold_peak_vram is not None) else "missing",
            "cold_load_steady_vram_mb": "measured" if steady_vram is not None else "missing",
            "kv_cache_capacity_tokens": "measured" if kv_capacity is not None else "missing",
            "ttft_p50_ms": "measured" if ttft_p50 is not None else "missing",
            "ttft_p95_ms": "measured" if ttft_p95 is not None else "missing",
            "decode_tokens_per_sec": "measured" if decode_tps is not None else "missing",
            "prompt_throughput_tokens_per_sec": "measured" if prompt_tps is not None else "missing",
            "post_unload_residual_vram_mb": "measured" if post_unload_residual is not None else "missing",
            "crashes_count": "measured",
            "nans_count": "measured",
            "has_memory_leak": "measured" if has_memory_leak is not None else "missing",
            "soak_duration_seconds": "measured",
        }

        ppl = None
        task_acc = None
        lc_acc = None
        quality_prov = None

        if quality_data:
            ppl = quality_data.get("perplexity")
            task_acc = quality_data.get("task_accuracy")
            lc_acc = quality_data.get("long_context_accuracy")
            quality_prov = quality_data.get("provenance")
            sources["perplexity"] = "imported_quality_evidence" if ppl is not None else "missing"
            sources["task_accuracy"] = "imported_quality_evidence" if task_acc is not None else "missing"
            sources["long_context_accuracy"] = "imported_quality_evidence" if lc_acc is not None else "missing"
        else:
            sources["perplexity"] = "missing"
            sources["task_accuracy"] = "missing"
            sources["long_context_accuracy"] = "missing"

        meas = ProfileMeasurements(
            idle_process_rss_mb=harness_rss,
            idle_gpu_memory_mb=idle_gpu_mem,
            model_load_duration_seconds=model_load_duration,
            cold_load_peak_vram_mb=cold_peak_vram if readiness_achieved else None,
            cold_load_steady_vram_mb=steady_vram,
            kv_cache_capacity_tokens=kv_capacity,
            ttft_p50_ms=ttft_p50,
            ttft_p95_ms=ttft_p95,
            decode_tokens_per_sec=decode_tps,
            prompt_throughput_tokens_per_sec=prompt_tps,
            perplexity=ppl,
            task_accuracy=task_acc,
            long_context_accuracy=lc_acc,
            crashes_count=total_crashes,
            nans_count=nans_count,
            has_memory_leak=has_memory_leak,
            post_unload_residual_vram_mb=post_unload_residual,
            soak_duration_seconds=soak_recorded_seconds,
            evidence_sources=sources,
            quality_provenance=quality_prov,
        )

        return meas, eval_notes

    def _check_http_readiness(self, port: int) -> bool:
        if self.http_client_factory:
            client = self.http_client_factory(port)
            return bool(getattr(client, "is_ready", lambda: True)())

        import urllib.request

        for endpoint in [f"http://127.0.0.1:{port}/health", f"http://127.0.0.1:{port}/v1/models"]:
            try:
                req = urllib.request.Request(endpoint, method="GET")
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
        return False

    def _send_soak_load_request(self, port: int) -> bool:
        """Issue a small generation request during soak to keep engine actively loaded."""
        if self.http_client_factory:
            client = self.http_client_factory(port)
            send_fn = getattr(client, "send_soak_load_request", None)
            if send_fn:
                return bool(send_fn())
            return True

        import urllib.request

        url = f"http://127.0.0.1:{port}/v1/completions"
        payload = json.dumps(
            {
                "prompt": "Soak active load test request.",
                "max_tokens": 16,
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _run_streaming_benchmark(
        self, port: int
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """
        Send streaming requests to /v1/completions with stream_options include_usage and calculate TTFT p50/p95 and throughput metrics.
        """
        if self.http_client_factory:
            client = self.http_client_factory(port)
            if hasattr(client, "run_streaming_benchmark"):
                return client.run_streaming_benchmark(
                    iterations=self.num_iterations, warmup=self.num_warmup
                )

        import urllib.request

        url = f"http://127.0.0.1:{port}/v1/completions"
        payload = json.dumps(
            {
                "prompt": "Benchmark prompt test string for TurboQuant KV cache evaluation.",
                "max_tokens": self.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}

        # Warmup requests
        for _ in range(self.num_warmup):
            try:
                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10.0) as resp:
                    _ = resp.read()
            except Exception:
                pass

        ttfts: list[float] = []
        decode_speeds: list[float] = []
        prompt_speeds: list[float] = []

        for _ in range(self.num_iterations):
            try:
                t0 = self.time_fn()
                chunks: list[tuple[float, str]] = []

                req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30.0) as resp:
                    for line in resp:
                        ts = self.time_fn()
                        line_str = line.decode("utf-8", errors="replace")
                        chunks.append((ts, line_str))

                ttft_ms, decode_sec, p_tokens, c_tokens = process_streaming_response(chunks, t0)
                if ttft_ms is not None:
                    ttfts.append(ttft_ms)

                if decode_sec is not None and c_tokens is not None and c_tokens > 1:
                    decode_tps = (c_tokens - 1) / decode_sec
                    decode_speeds.append(decode_tps)

                if ttft_ms is not None and p_tokens is not None:
                    ttft_sec = max(0.000001, ttft_ms / 1000.0)
                    prompt_speeds.append(p_tokens / ttft_sec)
            except Exception:
                pass

        if not ttfts or not prompt_speeds:
            return None, None, None, None

        ttft_p50 = calculate_percentile(ttfts, 50.0)
        ttft_p95 = calculate_percentile(ttfts, 95.0)
        avg_decode = sum(decode_speeds) / len(decode_speeds) if decode_speeds else None
        avg_prompt = sum(prompt_speeds) / len(prompt_speeds) if prompt_speeds else None

        return ttft_p50, ttft_p95, avg_decode, avg_prompt

    def _safely_teardown_process(self, proc: Any) -> None:
        """Gracefully terminate child process, wait bounded duration, then kill exact child."""
        try:
            poll_fn = getattr(proc, "poll", lambda: None)
            if poll_fn() is not None:
                return

            term_fn = getattr(proc, "terminate", None)
            if term_fn:
                term_fn()

            wait_fn = getattr(proc, "wait", None)
            if wait_fn:
                try:
                    wait_fn(timeout=self.shutdown_timeout_seconds)
                    return
                except (subprocess.TimeoutExpired, Exception):
                    pass

            kill_fn = getattr(proc, "kill", None)
            if kill_fn:
                kill_fn()

            if wait_fn:
                try:
                    wait_fn(timeout=5.0)
                except Exception:
                    pass
        except Exception:
            pass

    def _save_artifact_atomically(self, artifact: TurboQuantBenchmarkArtifact) -> None:
        """Atomically persist benchmark artifact to disk."""
        json_str = artifact.to_json(indent=2)
        tmp_path = self.artifact_output_path.with_suffix(".tmp")
        self.artifact_output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json_str, encoding="utf-8")
        tmp_path.replace(self.artifact_output_path)
