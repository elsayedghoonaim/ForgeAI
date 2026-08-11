"""
Pure deterministic unit tests for TurboQuant POC artifact model, evaluator, and hardware runner.

Verifies profile specifications, contract identity gating (vllm==0.22.1, CUDA >= 7.5, VRAM > 0),
preflight safety checks (Linux/WSL2, version match, executables, GPU), command construction,
dedicated used-memory GPU parsing, sampler failure vs genuine 0.0 MiB reading, valid streaming token timing,
completion_tokens > 1 decode rate requirement, active soak load with immediate stop on request failure,
quality evidence provenance persistence with mandatory consistent accuracy_scale ('percent' or 'fraction'),
memory leak thresholds relative to total VRAM, non-finite metric safety, observable error notes,
sequential runner lifecycle coordinator with fakes/mocks, and thin script plan/evaluate/execute CLI entrypoint behavior.

No vLLM, torch, GPU hardware, network, or process execution are required or loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

from scripts.benchmark_turboquant import main as script_main
from forgeai.benchmarking import (
    PROFILES_CATALOG,
    SCHEMA_VERSION,
    STATUS_FAIL,
    STATUS_INCOMPLETE,
    STATUS_NOT_RUN,
    STATUS_PASS,
    VLLM_CONTRACT_SPEC,
    VLLM_PINNED_VERSION,
    FutureCommandPlan,
    GateOutcome,
    HardwareMetadata,
    PreflightResult,
    ProfileMeasurements,
    ProfileResult,
    SequentialBenchmarkRunner,
    TurboQuantBenchmarkArtifact,
    calculate_percentile,
    create_benchmark_plan,
    default_gpu_memory_used_query,
    evaluate_artifact,
    load_quality_evidence,
    parse_cuda_compute_capability,
    parse_gpu_memory_used_output,
    parse_kv_cache_capacity_from_log,
    parse_log_anomalies,
    parse_sse_stream_chunk,
    process_streaming_response,
    run_preflight,
)


class TurboQuantContractAndCatalogTests(unittest.TestCase):
    """Test contract pinning, profile catalog rules, compute capability parser, and schema separation."""

    def test_vllm_pinning_contract(self) -> None:
        self.assertEqual(VLLM_PINNED_VERSION, "0.22.1")
        self.assertEqual(VLLM_CONTRACT_SPEC, "vllm==0.22.1")

        plan = create_benchmark_plan()
        self.assertEqual(plan.vllm_version, "vllm==0.22.1")
        self.assertEqual(plan.schema_version, SCHEMA_VERSION)

    def test_cuda_compute_capability_parser(self) -> None:
        self.assertEqual(parse_cuda_compute_capability((8, 0)), (8, 0))
        self.assertEqual(parse_cuda_compute_capability("8.0"), (8, 0))
        self.assertEqual(parse_cuda_compute_capability("7.5"), (7, 5))
        self.assertEqual(parse_cuda_compute_capability(8.9), (8, 9))
        self.assertIsNone(parse_cuda_compute_capability(None))
        self.assertIsNone(parse_cuda_compute_capability("invalid"))

    def test_profile_catalog_specifications(self) -> None:
        self.assertIn("auto", PROFILES_CATALOG)
        self.assertIn("fp8", PROFILES_CATALOG)
        self.assertIn("turboquant_4bit_nc", PROFILES_CATALOG)
        self.assertIn("turboquant_3bit_nc", PROFILES_CATALOG)

        auto_spec = PROFILES_CATALOG["auto"]
        self.assertEqual(auto_spec.expected_capacity_ratio, 1.0)
        self.assertEqual(auto_spec.dtype, "bfloat16")
        self.assertTrue(auto_spec.is_baseline)

        fp8_spec = PROFILES_CATALOG["fp8"]
        self.assertEqual(fp8_spec.expected_capacity_ratio, 2.0)
        self.assertEqual(fp8_spec.dtype, "bfloat16")
        self.assertFalse(fp8_spec.is_experimental)

        tq4_spec = PROFILES_CATALOG["turboquant_4bit_nc"]
        self.assertEqual(tq4_spec.expected_capacity_ratio, 3.8)
        self.assertEqual(tq4_spec.dtype, "bfloat16")
        self.assertTrue(tq4_spec.is_experimental)
        self.assertTrue(tq4_spec.is_poc_only)

        tq3_spec = PROFILES_CATALOG["turboquant_3bit_nc"]
        self.assertEqual(tq3_spec.expected_capacity_ratio, 4.9)
        self.assertEqual(tq3_spec.dtype, "bfloat16")
        self.assertTrue(tq3_spec.is_experimental)
        self.assertTrue(tq3_spec.is_poc_only)
        self.assertTrue(tq3_spec.is_aggressive)

    def test_shared_model_cache_and_distinct_work_dirs(self) -> None:
        plan = create_benchmark_plan(
            model="Qwen/Qwen3-0.6B",
            base_work_dir="/tmp/tq_bench_test",
            model_cache_dir="~/.forgeai/hf",
        )

        cache_dirs = set()
        work_dirs = set()

        for cp in plan.command_plans:
            self.assertEqual(cp.model_cache_dir, "~/.forgeai/hf")
            dl_idx = cp.argv.index("--download-dir")
            self.assertEqual(cp.argv[dl_idx + 1], "~/.forgeai/hf")
            cache_dirs.add(cp.argv[dl_idx + 1])
            work_dirs.add(cp.work_dir)

        self.assertEqual(len(cache_dirs), 1)
        self.assertEqual(len(work_dirs), 4)

    def test_controlled_bf16_engine_dtype_command_plans(self) -> None:
        plan = create_benchmark_plan(dtype="bfloat16", weight_quantization="none")

        for cp in plan.command_plans:
            self.assertEqual(cp.dtype, "bfloat16")
            self.assertIn("--dtype", cp.argv)
            dtype_idx = cp.argv.index("--dtype")
            self.assertEqual(cp.argv[dtype_idx + 1], "bfloat16")

        kv_dtypes = [cp.kv_cache_dtype for cp in plan.command_plans]
        self.assertEqual(kv_dtypes, ["auto", "fp8", "turboquant_4bit_nc", "turboquant_3bit_nc"])


class TurboQuantArtifactSerializationTests(unittest.TestCase):
    """Test blank plan generation, hardware validation flag defaults, and JSON roundtrips."""

    def test_blank_template_artifact_defaults(self) -> None:
        plan = create_benchmark_plan(model="Qwen/Qwen3-0.6B")
        self.assertEqual(plan.status, STATUS_NOT_RUN)
        self.assertFalse(plan.hardware_validation_performed)
        self.assertEqual(plan.model, "Qwen/Qwen3-0.6B")

        self.assertEqual(len(plan.command_plans), 4)
        for cp in plan.command_plans:
            self.assertIn("--dtype", cp.argv)
            self.assertIn("--kv-cache-dtype", cp.argv)
            self.assertIn(cp.kv_cache_dtype, cp.argv)
            self.assertEqual(cp.execution_mode, "sequential")

        evaluated = evaluate_artifact(plan)
        self.assertEqual(evaluated.status, STATUS_NOT_RUN)
        self.assertFalse(evaluated.hardware_validation_performed)
        contract_gate = evaluated.gate_outcomes.get("contract")
        self.assertIsNotNone(contract_gate)
        assert contract_gate is not None
        self.assertFalse(contract_gate.passed)
        self.assertEqual(contract_gate.status, STATUS_INCOMPLETE)

    def test_json_serialization_roundtrip(self) -> None:
        original = create_benchmark_plan(model="test/model-id")
        original.hardware = HardwareMetadata(
            device_name="NVIDIA RTX 4090",
            cuda_compute_capability="8.9",
            total_vram_mb=24576.0,
        )

        json_str = original.to_json()
        reconstructed = TurboQuantBenchmarkArtifact.from_json(json_str)

        self.assertEqual(reconstructed.model, original.model)
        self.assertEqual(reconstructed.vllm_version, original.vllm_version)
        self.assertEqual(reconstructed.hardware_validation_performed, original.hardware_validation_performed)
        self.assertEqual(reconstructed.hardware.device_name, "NVIDIA RTX 4090")
        self.assertEqual(reconstructed.hardware.total_vram_mb, 24576.0)
        self.assertEqual(len(reconstructed.command_plans), len(original.command_plans))


class TurboQuantContractIdentityGateTests(unittest.TestCase):
    """Test focused contract identity gate failures and missing evidence behavior."""

    def _create_base_valid_artifact(self) -> TurboQuantBenchmarkArtifact:
        hw = HardwareMetadata(
            device_name="NVIDIA A100-SXM4-80GB",
            cuda_compute_capability="8.0",
            total_vram_mb=81920.0,
            platform="cuda",
        )
        meas = ProfileMeasurements(
            idle_process_rss_mb=450.0,
            idle_gpu_memory_mb=1200.0,
            model_load_duration_seconds=5.0,
            cold_load_peak_vram_mb=18000.0,
            cold_load_steady_vram_mb=16000.0,
            kv_cache_capacity_tokens=1000,
            ttft_p50_ms=80.0,
            ttft_p95_ms=100.0,
            decode_tokens_per_sec=100.0,
            prompt_throughput_tokens_per_sec=1200.0,
            perplexity=10.0,
            task_accuracy=85.0,
            long_context_accuracy=90.0,
            crashes_count=0,
            nans_count=0,
            has_memory_leak=False,
            post_unload_residual_vram_mb=40.0,
            soak_duration_seconds=1800.0,
        )
        tq4_meas = ProfileMeasurements(
            idle_process_rss_mb=450.0,
            idle_gpu_memory_mb=1200.0,
            model_load_duration_seconds=4.5,
            cold_load_peak_vram_mb=12000.0,
            cold_load_steady_vram_mb=10000.0,
            kv_cache_capacity_tokens=3800,
            ttft_p50_ms=85.0,
            ttft_p95_ms=100.0,
            decode_tokens_per_sec=90.0,
            prompt_throughput_tokens_per_sec=1150.0,
            perplexity=10.2,
            task_accuracy=84.5,
            long_context_accuracy=89.2,
            crashes_count=0,
            nans_count=0,
            has_memory_leak=False,
            post_unload_residual_vram_mb=50.0,
            soak_duration_seconds=1800.0,
        )
        return TurboQuantBenchmarkArtifact(
            model="Qwen/Qwen3-0.6B",
            vllm_version=VLLM_CONTRACT_SPEC,
            hardware_validation_performed=True,
            status=STATUS_NOT_RUN,
            hardware=hw,
            profile_results={
                "auto": ProfileResult("auto", "auto", measurements=meas),
                "fp8": ProfileResult("fp8", "fp8", measurements=meas),
                "turboquant_4bit_nc": ProfileResult("turboquant_4bit_nc", "turboquant_4bit_nc", measurements=tq4_meas),
                "turboquant_3bit_nc": ProfileResult("turboquant_3bit_nc", "turboquant_3bit_nc", measurements=tq4_meas),
            },
        )

    def test_valid_contract_identity_passes(self) -> None:
        art = self._create_base_valid_artifact()
        evaluated = evaluate_artifact(art)

        contract_gate = evaluated.gate_outcomes.get("contract")
        self.assertIsNotNone(contract_gate)
        assert contract_gate is not None
        self.assertTrue(contract_gate.passed)
        self.assertEqual(contract_gate.status, STATUS_PASS)
        self.assertEqual(evaluated.status, STATUS_PASS)

    def test_absent_hardware_evidence_unvalidated_template(self) -> None:
        plan = create_benchmark_plan(model="Qwen/Qwen3-0.6B")
        evaluated = evaluate_artifact(plan)

        self.assertFalse(evaluated.hardware_validation_performed)
        self.assertEqual(evaluated.status, STATUS_NOT_RUN)

        contract_gate = evaluated.gate_outcomes["contract"]
        self.assertFalse(contract_gate.passed)
        self.assertEqual(contract_gate.status, STATUS_INCOMPLETE)
        self.assertIn("hardware_validation_performed", contract_gate.failed_metrics)
        self.assertIn("total_vram_mb", contract_gate.failed_metrics)
        self.assertIn("cuda_compute_capability", contract_gate.failed_metrics)

    def test_contract_failure_zero_vram_when_validation_claimed(self) -> None:
        art = self._create_base_valid_artifact()
        art.hardware_validation_performed = True
        art.hardware.total_vram_mb = 0.0

        evaluated = evaluate_artifact(art)
        self.assertEqual(evaluated.status, STATUS_FAIL)

        contract_gate = evaluated.gate_outcomes["contract"]
        self.assertFalse(contract_gate.passed)
        self.assertEqual(contract_gate.status, STATUS_FAIL)
        self.assertIn("total_vram_mb", contract_gate.failed_metrics)


class TurboQuantEvaluatorDeterministicTests(unittest.TestCase):
    """Test deterministic go/no-go evaluator decision gates relative to BF16 baseline."""

    def _create_synthetic_complete_artifact(
        self,
        hardware_validated: bool = True,
        tq4_capacity_tokens: int = 3800,
        tq4_ttft_p95_ms: float = 100.0,
        tq4_decode_tokens_per_sec: float = 90.0,
        tq4_perplexity: float = 10.2,
        tq4_task_accuracy: float = 84.5,
        tq4_long_context_accuracy: float = 89.2,
        tq4_crashes: int = 0,
        tq4_nans: int = 0,
        tq4_has_memory_leak: bool = False,
        tq4_post_unload_residual_vram_mb: float = 50.0,
        soak_duration_seconds: float = 1800.0,
    ) -> TurboQuantBenchmarkArtifact:
        """Helper creating synthetic complete benchmark artifact."""
        hw = HardwareMetadata(
            device_name="NVIDIA A100-SXM4-80GB",
            cuda_compute_capability="8.0",
            total_vram_mb=81920.0,
            platform="cuda",
        )

        auto_meas = ProfileMeasurements(
            idle_process_rss_mb=450.0,
            idle_gpu_memory_mb=1200.0,
            model_load_duration_seconds=5.0,
            cold_load_peak_vram_mb=18000.0,
            cold_load_steady_vram_mb=16000.0,
            kv_cache_capacity_tokens=1000,
            ttft_p50_ms=80.0,
            ttft_p95_ms=100.0,
            decode_tokens_per_sec=100.0,
            prompt_throughput_tokens_per_sec=1200.0,
            perplexity=10.0,
            task_accuracy=85.0,
            long_context_accuracy=90.0,
            crashes_count=0,
            nans_count=0,
            has_memory_leak=False,
            post_unload_residual_vram_mb=40.0,
            soak_duration_seconds=soak_duration_seconds,
        )

        tq4_meas = ProfileMeasurements(
            idle_process_rss_mb=450.0,
            idle_gpu_memory_mb=1200.0,
            model_load_duration_seconds=4.5,
            cold_load_peak_vram_mb=12000.0,
            cold_load_steady_vram_mb=10000.0,
            kv_cache_capacity_tokens=tq4_capacity_tokens,
            ttft_p50_ms=85.0,
            ttft_p95_ms=tq4_ttft_p95_ms,
            decode_tokens_per_sec=tq4_decode_tokens_per_sec,
            prompt_throughput_tokens_per_sec=1150.0,
            perplexity=tq4_perplexity,
            task_accuracy=tq4_task_accuracy,
            long_context_accuracy=tq4_long_context_accuracy,
            crashes_count=tq4_crashes,
            nans_count=tq4_nans,
            has_memory_leak=tq4_has_memory_leak,
            post_unload_residual_vram_mb=tq4_post_unload_residual_vram_mb,
            soak_duration_seconds=soak_duration_seconds,
        )

        tq3_meas = ProfileMeasurements(
            idle_process_rss_mb=450.0,
            idle_gpu_memory_mb=1200.0,
            model_load_duration_seconds=4.0,
            cold_load_peak_vram_mb=10000.0,
            cold_load_steady_vram_mb=8000.0,
            kv_cache_capacity_tokens=4900,
            ttft_p50_ms=90.0,
            ttft_p95_ms=120.0,
            decode_tokens_per_sec=80.0,
            prompt_throughput_tokens_per_sec=1100.0,
            perplexity=10.4,
            task_accuracy=83.5,
            long_context_accuracy=88.5,
            crashes_count=0,
            nans_count=0,
            has_memory_leak=False,
            post_unload_residual_vram_mb=50.0,
            soak_duration_seconds=soak_duration_seconds,
        )

        fp8_meas = ProfileMeasurements(
            idle_process_rss_mb=450.0,
            idle_gpu_memory_mb=1200.0,
            model_load_duration_seconds=4.8,
            cold_load_peak_vram_mb=15000.0,
            cold_load_steady_vram_mb=13000.0,
            kv_cache_capacity_tokens=2000,
            ttft_p50_ms=82.0,
            ttft_p95_ms=102.0,
            decode_tokens_per_sec=98.0,
            prompt_throughput_tokens_per_sec=1180.0,
            perplexity=10.05,
            task_accuracy=84.9,
            long_context_accuracy=89.9,
            crashes_count=0,
            nans_count=0,
            has_memory_leak=False,
            post_unload_residual_vram_mb=45.0,
            soak_duration_seconds=soak_duration_seconds,
        )

        artifact = TurboQuantBenchmarkArtifact(
            model="Qwen/Qwen3-0.6B",
            vllm_version=VLLM_CONTRACT_SPEC,
            hardware_validation_performed=hardware_validated,
            status=STATUS_NOT_RUN,
            hardware=hw,
            profile_results={
                "auto": ProfileResult("auto", "auto", measurements=auto_meas),
                "fp8": ProfileResult("fp8", "fp8", measurements=fp8_meas),
                "turboquant_4bit_nc": ProfileResult("turboquant_4bit_nc", "turboquant_4bit_nc", measurements=tq4_meas),
                "turboquant_3bit_nc": ProfileResult("turboquant_3bit_nc", "turboquant_3bit_nc", measurements=tq3_meas),
            },
        )
        return artifact

    def test_complete_passing_evidence(self) -> None:
        art = self._create_synthetic_complete_artifact(hardware_validated=True)
        evaluated = evaluate_artifact(art)

        self.assertEqual(evaluated.status, STATUS_PASS)
        self.assertTrue(evaluated.gate_outcomes["turboquant_4bit_nc"].passed)
        self.assertEqual(evaluated.gate_outcomes["turboquant_4bit_nc"].status, STATUS_PASS)

    def test_evaluator_rejects_nan_or_inf_measurements(self) -> None:
        art = self._create_synthetic_complete_artifact(hardware_validated=True)
        assert art.profile_results["turboquant_4bit_nc"].measurements is not None
        art.profile_results["turboquant_4bit_nc"].measurements.ttft_p95_ms = float("nan")

        evaluated = evaluate_artifact(art)
        self.assertNotEqual(evaluated.status, STATUS_PASS)
        self.assertEqual(evaluated.profile_results["turboquant_4bit_nc"].status, STATUS_INCOMPLETE)
        self.assertIn("ttft_p95_ms", evaluated.gate_outcomes["turboquant_4bit_nc"].failed_metrics)


class TurboQuantPreflightAndLogParsingTests(unittest.TestCase):
    """Test preflight safety checks, GPU memory parsing, log parsing, and streaming response processing."""

    def test_parse_gpu_memory_used_output(self) -> None:
        self.assertEqual(parse_gpu_memory_used_output("1234.5\n"), 1234.5)
        self.assertEqual(parse_gpu_memory_used_output("2048, MiB"), 2048.0)
        self.assertIsNone(parse_gpu_memory_used_output("invalid output"))

    def test_streaming_empty_initial_chunks(self) -> None:
        chunks = [
            (0.1, 'data: {"choices": [{"text": ""}]}'),
            (0.2, 'data: {"choices": [{"text": "Hello"}]}'),
            (0.3, 'data: {"choices": [{"text": " world"}]}'),
            (0.4, 'data: {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}'),
        ]
        t0 = 0.0
        ttft_ms, decode_sec, p_tok, c_tok = process_streaming_response(chunks, t0)
        self.assertIsNotNone(ttft_ms)
        self.assertAlmostEqual(ttft_ms, 200.0, places=2)  # First non-empty chunk at t=0.2 -> 200ms
        self.assertIsNotNone(decode_sec)
        self.assertEqual(p_tok, 10)
        self.assertEqual(c_tok, 5)

    def test_streaming_one_token_completion_leaves_decode_rate_unavailable(self) -> None:
        # One completion token (completion_tokens == 1) has no post-first-token decode tokens
        chunks = [
            (0.2, 'data: {"choices": [{"text": "A"}], "usage": {"prompt_tokens": 10, "completion_tokens": 1}}'),
        ]
        ttft_ms, decode_sec, p_tok, c_tok = process_streaming_response(chunks, 0.0)
        self.assertIsNotNone(ttft_ms)
        self.assertIsNone(decode_sec)  # Decode rate unavailable when completion_tokens <= 1
        self.assertEqual(p_tok, 10)
        self.assertEqual(c_tok, 1)

    def test_streaming_missing_usage(self) -> None:
        chunks = [
            (0.1, 'data: {"choices": [{"text": "Hello"}]}'),
        ]
        ttft_ms, decode_sec, p_tok, c_tok = process_streaming_response(chunks, 0.0)
        self.assertIsNotNone(ttft_ms)
        self.assertIsNone(decode_sec)
        self.assertIsNone(p_tok)
        self.assertIsNone(c_tok)

    def test_calculate_percentile_deterministic(self) -> None:
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(calculate_percentile(data, 50.0), 30.0)
        self.assertEqual(calculate_percentile(data, 0.0), 10.0)
        self.assertEqual(calculate_percentile(data, 100.0), 50.0)

    def test_parse_kv_cache_capacity_from_log(self) -> None:
        log_sample = (
            "INFO 08-10 12:00:00 [engine.py:123] Capturing CUDA graph...\n"
            "INFO 08-10 12:00:01 [kv_cache.py:45] GPU KV cache size: 123,456 tokens\n"
        )
        self.assertEqual(parse_kv_cache_capacity_from_log(log_sample), 123456)


class TurboQuantQualityEvidenceTests(unittest.TestCase):
    """Test loading and validation of quality evidence JSON artifacts."""

    def test_load_valid_quality_evidence_percent_scale(self) -> None:
        evidence_dict = {
            "provenance": {
                "dataset": "GSM8K + NeedleInAHaystack",
                "evaluator": "ForgeAI Quality Evaluator v1.0",
                "accuracy_scale": "percent",
            },
            "profiles": {
                "auto": {"perplexity": 10.0, "task_accuracy": 85.0, "long_context_accuracy": 90.0},
                "fp8": {"perplexity": 10.05, "task_accuracy": 84.9, "long_context_accuracy": 89.9},
                "turboquant_4bit_nc": {"perplexity": 10.2, "task_accuracy": 84.5, "long_context_accuracy": 89.2},
                "turboquant_3bit_nc": {"perplexity": 10.4, "task_accuracy": 83.5, "long_context_accuracy": 88.5},
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = Path(tmp_dir) / "quality.json"
            fpath.write_text(json.dumps(evidence_dict), encoding="utf-8")

            loaded = load_quality_evidence(fpath)
            self.assertIn("auto", loaded)
            self.assertEqual(loaded["auto"]["perplexity"], 10.0)
            self.assertEqual(loaded["auto"]["task_accuracy"], 85.0)
            self.assertEqual(loaded["auto"]["provenance"]["accuracy_scale"], "percent")

    def test_load_valid_quality_evidence_fraction_scale_normalized(self) -> None:
        evidence_dict = {
            "provenance": {
                "dataset": "GSM8K",
                "evaluator": "ForgeAI Evaluator",
                "accuracy_scale": "fraction",
            },
            "profiles": {
                "auto": {"perplexity": 10.0, "task_accuracy": 0.85, "long_context_accuracy": 0.90},
                "fp8": {"perplexity": 10.05, "task_accuracy": 0.849, "long_context_accuracy": 0.899},
                "turboquant_4bit_nc": {"perplexity": 10.2, "task_accuracy": 0.845, "long_context_accuracy": 0.892},
                "turboquant_3bit_nc": {"perplexity": 10.4, "task_accuracy": 0.835, "long_context_accuracy": 0.885},
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = Path(tmp_dir) / "quality_fraction.json"
            fpath.write_text(json.dumps(evidence_dict), encoding="utf-8")

            loaded = load_quality_evidence(fpath)
            self.assertEqual(loaded["auto"]["task_accuracy"], 85.0)  # Normalized to percentage points
            self.assertEqual(loaded["auto"]["long_context_accuracy"], 90.0)

    def test_load_quality_evidence_rejects_missing_accuracy_scale(self) -> None:
        evidence_dict = {
            "provenance": {"dataset": "GSM8K", "evaluator": "ForgeAI Evaluator"},
            "profiles": {
                "auto": {"perplexity": 10.0, "task_accuracy": 85.0, "long_context_accuracy": 90.0},
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = Path(tmp_dir) / "no_scale.json"
            fpath.write_text(json.dumps(evidence_dict), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_quality_evidence(fpath)
            self.assertIn("accuracy_scale", str(ctx.exception).lower())

    def test_load_quality_evidence_rejects_inconsistent_mixed_scales(self) -> None:
        evidence_dict = {
            "profiles": {
                "auto": {
                    "perplexity": 10.0,
                    "task_accuracy": 85.0,
                    "long_context_accuracy": 90.0,
                    "provenance": {"dataset": "D", "evaluator": "E", "accuracy_scale": "percent"},
                },
                "fp8": {
                    "perplexity": 10.0,
                    "task_accuracy": 0.85,
                    "long_context_accuracy": 0.90,
                    "provenance": {"dataset": "D", "evaluator": "E", "accuracy_scale": "fraction"},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = Path(tmp_dir) / "mixed_scale.json"
            fpath.write_text(json.dumps(evidence_dict), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_quality_evidence(fpath)
            self.assertIn("inconsistent accuracy_scale", str(ctx.exception).lower())

    def test_load_quality_evidence_rejects_out_of_bounds_values_for_scale(self) -> None:
        evidence_dict = {
            "provenance": {"dataset": "D", "evaluator": "E", "accuracy_scale": "fraction"},
            "profiles": {
                "auto": {"perplexity": 10.0, "task_accuracy": 1.5, "long_context_accuracy": 0.90},  # 1.5 > 1.0 for fraction
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            fpath = Path(tmp_dir) / "oob.json"
            fpath.write_text(json.dumps(evidence_dict), encoding="utf-8")

            with self.assertRaises(ValueError) as ctx:
                load_quality_evidence(fpath)
            self.assertIn("out of bounds", str(ctx.exception).lower())


class TurboQuantSequentialRunnerLifecycleTests(unittest.TestCase):
    """Test sequential execution lifecycle, sampler exception safety, active soak load failure, and memory leak calculation."""

    class FakeProcess:
        def __init__(self, argv: list[str], stdout: Any, stderr: Any, shell: bool = False) -> None:
            self.argv = argv
            self.stdout = stdout
            self.stderr = stderr
            self.shell = shell
            self.terminated = False
            self.killed = False
            self._poll_return = None
            if stdout:
                stdout.write("INFO: Server starting...\nGPU KV cache size: 50,000 tokens\n")
                stdout.flush()

        def poll(self) -> int | None:
            return self._poll_return

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            self._poll_return = 0
            return 0

        def kill(self) -> None:
            self.killed = True
            self._poll_return = -9

    class FakeHttpClient:
        def __init__(self, port: int, soak_fails: bool = False) -> None:
            self.port = port
            self.is_ready = lambda: True
            self.soak_requests_sent = 0
            self.soak_fails = soak_fails

        def send_soak_load_request(self) -> bool:
            self.soak_requests_sent += 1
            return not self.soak_fails

        def run_streaming_benchmark(self, iterations: int, warmup: int) -> tuple[float, float, float, float]:
            return 80.0, 100.0, 95.0, 1100.0

    def test_gpu_sampler_failure_marks_evidence_missing_and_fails_eval(self) -> None:
        plan = create_benchmark_plan()
        hw_sample = HardwareMetadata(
            device_name="NVIDIA A100-SXM4-80GB",
            cuda_compute_capability="8.0",
            total_vram_mb=81920.0,
            platform="cuda",
        )

        def failing_gpu_sampler() -> float | None:
            return None  # Sampler query failure

        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = SequentialBenchmarkRunner(
                plan_artifact=plan,
                output_dir=Path(tmp_dir) / "out",
                soak_duration_seconds=0.0,
                acknowledge_hardware_run=True,
                preflight_fn=lambda: PreflightResult(passed=True, errors=[], hardware=hw_sample),
                process_factory=lambda argv, stdout, stderr, shell=False: self.FakeProcess(argv, stdout, stderr, shell),
                http_client_factory=lambda port: self.FakeHttpClient(port),
                gpu_sampler_fn=failing_gpu_sampler,
                rss_fn=lambda: 450.0,
                sleep_fn=lambda sec: None,
            )

            result = runner.run()
            # Verify evidence source is marked "missing" (NOT "measured") and result cannot pass
            for prof in result.profile_results.values():
                assert prof.measurements is not None
                self.assertIsNone(prof.measurements.idle_gpu_memory_mb)
                self.assertEqual(prof.measurements.evidence_sources["idle_gpu_memory_mb"], "missing")
                self.assertNotEqual(prof.status, STATUS_PASS)

    def test_failed_soak_load_stops_soak_accounting_immediately(self) -> None:
        plan = create_benchmark_plan()
        hw_sample = HardwareMetadata(
            device_name="NVIDIA A100-SXM4-80GB",
            cuda_compute_capability="8.0",
            total_vram_mb=81920.0,
            platform="cuda",
        )

        sim_time = [0.0]

        def fake_time() -> float:
            return sim_time[0]

        def fake_sleep(sec: float) -> None:
            sim_time[0] += sec

        with tempfile.TemporaryDirectory() as tmp_dir:
            client = self.FakeHttpClient(11435, soak_fails=True)

            runner = SequentialBenchmarkRunner(
                plan_artifact=plan,
                output_dir=Path(tmp_dir) / "out",
                readiness_timeout_seconds=5.0,
                soak_duration_seconds=1800.0,
                acknowledge_hardware_run=True,
                preflight_fn=lambda: PreflightResult(passed=True, errors=[], hardware=hw_sample),
                process_factory=lambda argv, stdout, stderr, shell=False: self.FakeProcess(argv, stdout, stderr, shell),
                http_client_factory=lambda port: client,
                gpu_sampler_fn=lambda: 1200.0,
                rss_fn=lambda: 450.0,
                sleep_fn=fake_sleep,
                time_fn=fake_time,
            )

            result = runner.run()
            auto_meas = result.profile_results["auto"].measurements
            assert auto_meas is not None
            # Verify soak duration stopped immediately after failed request (< 1800s) and profile failed stability gate
            self.assertLess(auto_meas.soak_duration_seconds, 1800.0)
            self.assertNotEqual(result.status, STATUS_PASS)

    def test_memory_leak_threshold_uses_total_vram_not_idle_used(self) -> None:
        plan = create_benchmark_plan()
        hw_sample = HardwareMetadata(
            device_name="NVIDIA A100-SXM4-80GB",
            cuda_compute_capability="8.0",
            total_vram_mb=81920.0,
            platform="cuda",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            sampler_calls = [1200.0, 1200.0, 1200.0, 1700.0]  # post_unload = 1700, baseline = 1200 -> residual = 500
            call_idx = [0]

            def fake_sampler() -> float | None:
                idx = min(call_idx[0], len(sampler_calls) - 1)
                val = sampler_calls[idx]
                call_idx[0] += 1
                return val

            runner = SequentialBenchmarkRunner(
                plan_artifact=plan,
                output_dir=Path(tmp_dir) / "out",
                soak_duration_seconds=0.0,
                acknowledge_hardware_run=True,
                preflight_fn=lambda: PreflightResult(passed=True, errors=[], hardware=hw_sample),
                process_factory=lambda argv, stdout, stderr, shell=False: self.FakeProcess(argv, stdout, stderr, shell),
                http_client_factory=lambda port: self.FakeHttpClient(port),
                gpu_sampler_fn=fake_sampler,
                rss_fn=lambda: 450.0,
                sleep_fn=lambda sec: None,
            )

            result = runner.run()
            for prof in result.profile_results.values():
                assert prof.measurements is not None
                self.assertFalse(prof.measurements.has_memory_leak)


class TurboQuantExecuteCLITests(unittest.TestCase):
    """Test CLI subcommand safety and argument validation."""

    def test_execute_without_acknowledgement_exits_before_preflight(self) -> None:
        code = script_main(["execute", "--model", "Qwen/Qwen3-0.6B"])
        self.assertEqual(code, 1)

    def test_execute_invalid_nonpositive_ranges_rejected(self) -> None:
        code_iter = script_main(["execute", "--acknowledge-hardware-run", "--iterations", "0"])
        self.assertEqual(code_iter, 1)

        code_port = script_main(["execute", "--acknowledge-hardware-run", "--base-port", "-1"])
        self.assertEqual(code_port, 1)


if __name__ == "__main__":
    unittest.main()
