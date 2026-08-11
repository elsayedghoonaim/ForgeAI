"""
Pure deterministic unit tests for resource profiles, memory estimator, metrics, and CLI profile save.

No vLLM, torch, GPU hardware, network, process execution, or benchmarks are loaded or used.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from forgeai.cli.commands.profile import app as profile_app
from forgeai.core.resource_profiles import (
    PRIMARY_PROFILES,
    RESOURCE_PROFILES,
    ProfileUnavailableError,
    UnsupportedPlatformError,
    get_resource_profile,
    select_resource_profile,
)
from forgeai.monitoring.metrics import (
    generate_metrics,
    record_admission_rejection,
    record_engine_eviction,
    record_engine_state,
    record_model_load,
    record_oom_event,
    record_ttft,
    update_decode_throughput,
    update_gpu_metrics,
    update_kv_cache_metrics,
    update_process_rss,
    update_queued_requests,
)
from forgeai.utils.helpers import load_yaml
from forgeai.utils.memory_estimator import estimate_from_preset, estimate_vram, print_estimate

runner = CliRunner()


class ResourceProfileCatalogTests(unittest.TestCase):
    """Test resource profile catalog facts and lookup helper."""

    def test_primary_profiles_exist(self) -> None:
        self.assertIn("auto", RESOURCE_PROFILES)
        self.assertIn("fp8", RESOURCE_PROFILES)
        self.assertIn("turboquant_4bit_nc", RESOURCE_PROFILES)
        self.assertIn("turboquant_3bit_nc", RESOURCE_PROFILES)
        self.assertIn("turboquant_k8v4", RESOURCE_PROFILES)

        self.assertEqual(PRIMARY_PROFILES, ("auto", "fp8", "turboquant_4bit_nc", "turboquant_3bit_nc"))

    def test_catalog_facts(self) -> None:
        auto_prof = RESOURCE_PROFILES["auto"]
        self.assertEqual(auto_prof.expected_capacity_ratio, 1.0)
        self.assertEqual(auto_prof.bytes_per_element, 2.0)
        self.assertFalse(auto_prof.is_experimental)
        self.assertFalse(auto_prof.is_poc_gated)

        fp8_prof = RESOURCE_PROFILES["fp8"]
        self.assertEqual(fp8_prof.expected_capacity_ratio, 2.0)
        self.assertEqual(fp8_prof.bytes_per_element, 1.0)
        self.assertTrue(fp8_prof.requires_validation)

        tq4_prof = RESOURCE_PROFILES["turboquant_4bit_nc"]
        self.assertEqual(tq4_prof.expected_capacity_ratio, 3.8)
        self.assertTrue(tq4_prof.is_experimental)
        self.assertTrue(tq4_prof.is_poc_gated)
        self.assertEqual(tq4_prof.min_cuda_compute_capability, (7, 5))

        tq3_prof = RESOURCE_PROFILES["turboquant_3bit_nc"]
        self.assertEqual(tq3_prof.expected_capacity_ratio, 4.9)
        self.assertTrue(tq3_prof.is_experimental)
        self.assertTrue(tq3_prof.is_poc_gated)
        self.assertEqual(tq3_prof.risk_level, "high")

    def test_lookup_aliases_and_case_insensitivity(self) -> None:
        self.assertEqual(get_resource_profile("auto").name, "auto")
        self.assertEqual(get_resource_profile("bf16").name, "auto")
        self.assertEqual(get_resource_profile("BFLOAT16").name, "auto")
        self.assertEqual(get_resource_profile("float16").name, "auto")
        self.assertEqual(get_resource_profile("FP8").name, "fp8")
        self.assertEqual(get_resource_profile("TURBOQUANT_4BIT_NC").name, "turboquant_4bit_nc")

    def test_unknown_profile_lookup_raises(self) -> None:
        with self.assertRaises(ProfileUnavailableError):
            get_resource_profile("invalid_profile_xyz")


class ResourceProfileSelectionTests(unittest.TestCase):
    """Test selection policies, conservative defaults, platform policy, and no-fallback errors."""

    def test_conservative_default_selection(self) -> None:
        # Default workload yields auto (BF16) without needing hardware capability
        prof_default = select_resource_profile(platform="cuda", workload="default")
        self.assertEqual(prof_default.name, "auto")

        prof_latency = select_resource_profile(platform="cuda", workload="latency")
        self.assertEqual(prof_latency.name, "auto")

        prof_quality = select_resource_profile(platform="cuda", workload="quality")
        self.assertEqual(prof_quality.name, "auto")

    def test_unknown_workload_raises_value_error(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            select_resource_profile(platform="cuda", workload="invalid_workload_abc")
        self.assertIn("unknown or unsupported workload", str(ctx.exception).lower())

    def test_unsupported_platform_cpu_raises_error(self) -> None:
        with self.assertRaises(UnsupportedPlatformError) as ctx:
            select_resource_profile(platform="cpu")
        self.assertIn("unsupported", str(ctx.exception).lower())

        with self.assertRaises(UnsupportedPlatformError):
            select_resource_profile(platform="intel")

        with self.assertRaises(UnsupportedPlatformError):
            select_resource_profile(platform="metal")

    def test_rocm_never_selects_turboquant(self) -> None:
        # Memory workload on ROCm selects fp8 if allowed, otherwise auto
        prof_rocm_mem = select_resource_profile(platform="rocm", workload="memory", allow_fp8=True)
        self.assertEqual(prof_rocm_mem.name, "fp8")

        prof_rocm_no_fp8 = select_resource_profile(platform="rocm", workload="memory", allow_fp8=False)
        self.assertEqual(prof_rocm_no_fp8.name, "auto")

        # Explicitly requesting TurboQuant on ROCm raises UnsupportedPlatformError
        with self.assertRaises(UnsupportedPlatformError):
            select_resource_profile(platform="rocm", requested_profile="turboquant_4bit_nc")

    def test_missing_cuda_cc_cannot_enable_turboquant(self) -> None:
        # Default cuda_compute_capability is None; explicit request fails CC gate
        with self.assertRaises(ProfileUnavailableError) as ctx:
            select_resource_profile(
                platform="cuda",
                cuda_compute_capability=None,
                requested_profile="turboquant_4bit_nc",
                hw_validation_passed=True,
                quality_validation_passed=True,
            )
        self.assertIn("compute capability", str(ctx.exception).lower())

        # Auto selection with memory workload and missing CC cannot select TurboQuant
        prof_no_cc = select_resource_profile(
            platform="cuda",
            cuda_compute_capability=None,
            workload="memory",
            hw_validation_passed=True,
            quality_validation_passed=True,
            allow_fp8=True,
        )
        self.assertEqual(prof_no_cc.name, "fp8")

    def test_nvidia_memory_workload_selection_with_cc(self) -> None:
        # NVIDIA CUDA memory workload with CC >= 7.5 and validation flags true selects turboquant_4bit_nc
        prof_tq4 = select_resource_profile(
            platform="cuda",
            cuda_compute_capability=(8, 0),
            workload="memory",
            hw_validation_passed=True,
            quality_validation_passed=True,
        )
        self.assertEqual(prof_tq4.name, "turboquant_4bit_nc")

        # NVIDIA CUDA memory workload without validation flags, with allow_fp8 selects fp8
        prof_fp8 = select_resource_profile(
            platform="cuda",
            cuda_compute_capability=(8, 0),
            workload="memory",
            allow_fp8=True,
            hw_validation_passed=False,
        )
        self.assertEqual(prof_fp8.name, "fp8")

    def test_turboquant_compute_capability_check(self) -> None:
        # CUDA compute capability 7.0 (< 7.5) requesting TurboQuant 4-bit raises ProfileUnavailableError
        with self.assertRaises(ProfileUnavailableError) as ctx:
            select_resource_profile(
                platform="cuda",
                cuda_compute_capability=(7, 0),
                requested_profile="turboquant_4bit_nc",
                hw_validation_passed=True,
                quality_validation_passed=True,
            )
        self.assertIn("compute capability", str(ctx.exception).lower())

    def test_poc_gated_turboquant_requires_validation_flags(self) -> None:
        # TurboQuant 4-bit without validation flags raises ProfileUnavailableError
        with self.assertRaises(ProfileUnavailableError):
            select_resource_profile(
                platform="cuda",
                cuda_compute_capability=(8, 0),
                requested_profile="turboquant_4bit_nc",
                hw_validation_passed=False,
            )

        # TurboQuant 3-bit requires aggressive_quality_passed
        with self.assertRaises(ProfileUnavailableError):
            select_resource_profile(
                platform="cuda",
                cuda_compute_capability=(8, 0),
                requested_profile="turboquant_3bit_nc",
                hw_validation_passed=True,
                quality_validation_passed=True,
                aggressive_quality_passed=False,
            )

        # TurboQuant 3-bit with all validation flags succeeds
        prof_tq3 = select_resource_profile(
            platform="cuda",
            cuda_compute_capability=(8, 0),
            requested_profile="turboquant_3bit_nc",
            hw_validation_passed=True,
            quality_validation_passed=True,
            aggressive_quality_passed=True,
        )
        self.assertEqual(prof_tq3.name, "turboquant_3bit_nc")


class KVMemoryEstimatorTests(unittest.TestCase):
    """Test theoretical KV-cache memory estimator calculation, ratio scaling, and unknown dtype rejection."""

    def test_kv_cache_ratio_scaling(self) -> None:
        est_auto = estimate_vram(param_count_billions=7.0, kv_cache_dtype="auto")
        est_fp8 = estimate_vram(param_count_billions=7.0, kv_cache_dtype="fp8")
        est_tq4 = estimate_vram(param_count_billions=7.0, kv_cache_dtype="turboquant_4bit_nc")
        est_tq3 = estimate_vram(param_count_billions=7.0, kv_cache_dtype="turboquant_3bit_nc")

        # FP8 capacity ratio is 2.0x -> KV cache MB should be half of auto
        self.assertAlmostEqual(est_fp8.kv_cache_mb, est_auto.kv_cache_mb / 2.0, places=4)

        # TurboQuant 4-bit NC ratio is 3.8x
        self.assertAlmostEqual(est_tq4.kv_cache_mb, est_auto.kv_cache_mb / 3.8, places=4)

        # TurboQuant 3-bit NC ratio is 4.9x
        self.assertAlmostEqual(est_tq3.kv_cache_mb, est_auto.kv_cache_mb / 4.9, places=4)

    def test_vram_estimate_extended_fields(self) -> None:
        est = estimate_vram(param_count_billions=7.0, kv_cache_dtype="fp8")
        self.assertEqual(est.kv_cache_dtype, "fp8")
        self.assertEqual(est.kv_capacity_ratio, 2.0)
        self.assertTrue(est.is_theoretical_estimate)
        self.assertGreater(est.bytes_per_kv_token, 0.0)

    def test_resource_policy_defaults(self) -> None:
        est = estimate_vram(param_count_billions=7.0)
        # Default available_vram_mb should be 24000
        self.assertEqual(est.available_vram_mb, 24000)

    def test_unknown_kv_dtype_rejection(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            estimate_vram(param_count_billions=7.0, kv_cache_dtype="invalid_kv_dtype")
        self.assertIn("unknown or unsupported kv cache dtype", str(ctx.exception).lower())

    def test_preset_estimation_with_kv_dtype(self) -> None:
        est = estimate_from_preset("llama-7b", kv_cache_dtype="turboquant_4bit_nc")
        self.assertIsNotNone(est)
        assert est is not None
        self.assertEqual(est.kv_cache_dtype, "turboquant_4bit_nc")
        self.assertEqual(est.kv_capacity_ratio, 3.8)

    def test_print_estimate_runs_without_error(self) -> None:
        est = estimate_vram(param_count_billions=7.0, kv_cache_dtype="auto")
        # Ensure printing does not raise any formatting errors
        with patch("forgeai.utils.memory_estimator.console.print") as mock_print:
            print_estimate(est, "Llama-7B")
            mock_print.assert_called_once()


class ResourceMetricsTests(unittest.TestCase):
    """Test low-cardinality resource metrics helpers and label validation."""

    def test_bounded_label_validation(self) -> None:
        # Invalid engine state falls back to "other"
        record_engine_state(state="invalid_state_xyz", count=1)
        record_engine_eviction(reason="unbounded_reason_abc")
        record_admission_rejection(reason="unbounded_rejection_123")

        # Valid calls
        record_engine_state(state="ready", count=2)
        record_engine_eviction(reason="vram_pressure")
        record_admission_rejection(reason="queue_full")

        metrics_text = generate_metrics()
        self.assertIn("forgeai_engines_by_state", metrics_text)
        self.assertIn("forgeai_engine_evictions_total", metrics_text)
        self.assertIn("forgeai_admission_rejections_total", metrics_text)

    def test_resource_update_helpers(self) -> None:
        update_process_rss(1024 * 1024 * 500)  # 500 MB
        update_gpu_metrics(gpu_id=0, used_bytes=4e9, total_bytes=8e9, util_pct=50.0, free_bytes=4e9, reserved_bytes=4.5e9)
        record_model_load(duration_seconds=12.5, peak_memory_bytes=6e9)
        update_kv_cache_metrics(capacity_bytes=2e9, usage_bytes=1e9)
        update_queued_requests(3)
        record_ttft(0.045)
        update_decode_throughput(85.5)
        record_oom_event()

        metrics_text = generate_metrics()
        self.assertIn("forgeai_process_rss_bytes", metrics_text)
        self.assertIn("forgeai_gpu_memory_free_bytes", metrics_text)
        self.assertIn("forgeai_model_load_duration_seconds", metrics_text)
        self.assertIn("forgeai_kv_cache_utilization_ratio", metrics_text)
        self.assertIn("forgeai_queued_requests", metrics_text)
        self.assertIn("forgeai_time_to_first_token_seconds", metrics_text)
        self.assertIn("forgeai_decode_throughput_tokens_per_second", metrics_text)
        self.assertIn("forgeai_oom_events_total", metrics_text)


class SavedProfileDefaultsTests(unittest.TestCase):
    """Test CLI profile save default values, schema separation, and --kv-cache-dtype validation."""

    def test_save_profile_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_settings = patch("forgeai.core.config.DevToolSettings", return_value=MagicMock(profiles_dir=tmp_dir))
            with mock_settings:
                result = runner.invoke(profile_app, ["save", "test_default_profile", "--model", "meta-llama/Llama-2-7b"])
                self.assertEqual(result.exit_code, 0)

                saved_file = Path(tmp_dir) / "test_default_profile.yaml"
                self.assertTrue(saved_file.exists())

                data = load_yaml(saved_file)
                self.assertEqual(data["gpu"]["gpu_memory_utilization"], 0.85)
                self.assertEqual(data["model"]["max_num_seqs"], 4)
                self.assertEqual(data["server"]["host"], "127.0.0.1")
                self.assertEqual(data["server"]["port"], 11434)
                self.assertEqual(data["kv_cache"]["dtype"], "auto")
                self.assertNotIn("kv_cache_dtype", data["model"])

    def test_save_profile_with_fp8_kv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_settings = patch("forgeai.core.config.DevToolSettings", return_value=MagicMock(profiles_dir=tmp_dir))
            with mock_settings:
                result = runner.invoke(
                    profile_app,
                    ["save", "test_fp8_profile", "--model", "meta-llama/Llama-2-7b", "--kv-cache-dtype", "fp8"],
                )
                self.assertEqual(result.exit_code, 0)

                saved_file = Path(tmp_dir) / "test_fp8_profile.yaml"
                data = load_yaml(saved_file)
                self.assertEqual(data["kv_cache"]["dtype"], "fp8")
                self.assertNotIn("kv_cache_dtype", data["model"])

    def test_save_profile_rejects_aggressive_3bit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_settings = patch("forgeai.core.config.DevToolSettings", return_value=MagicMock(profiles_dir=tmp_dir))
            with mock_settings:
                result = runner.invoke(
                    profile_app,
                    ["save", "test_3bit_profile", "--model", "meta-llama/Llama-2-7b", "--kv-cache-dtype", "turboquant_3bit_nc"],
                )
                self.assertEqual(result.exit_code, 1)
                self.assertIn("turboquant_3bit_nc", result.output.lower())


if __name__ == "__main__":
    unittest.main()
