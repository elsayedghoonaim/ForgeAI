"""Regression tests for Phase 0 remediation fixes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forgeai.benchmarking.runner import load_quality_evidence
from forgeai.benchmarking.turboquant import (
    STATUS_NOT_RUN,
    ProfileMeasurements,
    create_benchmark_plan,
    evaluate_artifact,
)
from forgeai.core.backends.vllm_backend import VLLMBackend
from forgeai.core.config import DevToolSettings
from forgeai.core.engine import EngineKey, EngineManager
from forgeai.models.loader import CacheManager
from forgeai.models.manifest import validate_repo_id_string


class TestRepoIdValidation:
    @pytest.mark.parametrize(
        "bad",
        [
            "a" * 201,
            "owner/" + "a" * 97,
            "owner/my--model",
            " owner/model",
            "owner/model ",
            "owner/model\n",
            "",
            "   ",
        ],
    )
    def test_rejected_with_single_prefix(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Invalid HuggingFace repository ID format"):
            validate_repo_id_string(bad)

    def test_limits_are_inclusive(self) -> None:
        validate_repo_id_string("a" * 96 + "/" + "b" * 96)
        validate_repo_id_string("a" * 96)


class TestTurboQuantEvaluation:
    def test_auto_profile_without_measurements_is_not_run(self) -> None:
        art = evaluate_artifact(create_benchmark_plan())
        assert art.profile_results["auto"].status == STATUS_NOT_RUN
        assert art.gate_outcomes["auto"].status == STATUS_NOT_RUN

    def test_zero_baseline_values_do_not_crash(self) -> None:
        def meas(**over: float) -> ProfileMeasurements:
            base = dict(
                idle_process_rss_mb=1.0,
                idle_gpu_memory_mb=1.0,
                model_load_duration_seconds=1.0,
                cold_load_peak_vram_mb=1.0,
                cold_load_steady_vram_mb=1.0,
                kv_cache_capacity_tokens=100,
                ttft_p50_ms=1.0,
                ttft_p95_ms=1.0,
                decode_tokens_per_sec=1.0,
                prompt_throughput_tokens_per_sec=1.0,
                perplexity=5.0,
                task_accuracy=50.0,
                long_context_accuracy=50.0,
                crashes_count=0,
                nans_count=0,
                has_memory_leak=False,
                post_unload_residual_vram_mb=0.0,
                soak_duration_seconds=1800.0,
            )
            base.update(over)
            return ProfileMeasurements(**base)  # type: ignore[arg-type]

        art = create_benchmark_plan()
        art.hardware_validation_performed = True
        art.hardware.cuda_compute_capability = (8, 0)
        art.hardware.total_vram_mb = 40000.0
        zero = meas(
            kv_cache_capacity_tokens=0, ttft_p95_ms=0.0, decode_tokens_per_sec=0.0
        )
        for name in art.profile_results:
            art.profile_results[name].measurements = meas()
        art.profile_results["auto"].measurements = zero
        art.profile_results["fp8"].measurements = meas()

        result = evaluate_artifact(art)  # must not raise ZeroDivisionError
        assert not result.gate_outcomes["turboquant_4bit_nc"].passed


class TestRayShutdownOwnership:
    def test_backend_shutdown_does_not_touch_ray(self) -> None:
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True
        backend = VLLMBackend(DevToolSettings(model_name="dummy"))
        backend._engine = MagicMock()
        backend._is_running = True
        with patch.dict("sys.modules", {"ray": mock_ray}):
            backend.shutdown()
        mock_ray.shutdown.assert_not_called()

    def test_manager_shuts_ray_down_only_when_no_engines_remain(self) -> None:
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True

        async def factory(key: EngineKey) -> MagicMock:
            return MagicMock()

        async def run() -> None:
            manager = EngineManager(engine_factory=factory, max_loaded_models=2)
            k1, k2 = EngineKey(repo_id="m/one"), EngineKey(repo_id="m/two")
            await manager.acquire(k1)
            await manager.acquire(k2)
            await manager.release(k1)
            await manager.release(k2)
            await manager.stop(k1)
            mock_ray.shutdown.assert_not_called()
            await manager.shutdown()

        with patch.dict("sys.modules", {"ray": mock_ray}):
            asyncio.run(run())
        mock_ray.shutdown.assert_called_once()


class TestSnapshotLookup:
    def test_single_unrelated_snapshot_is_not_returned(self, tmp_path: Path) -> None:
        cache = CacheManager(forgeai_home=tmp_path)
        repo = cache.get_repo_dir("owner/model")
        (repo / "snapshots" / "abc123").mkdir(parents=True)
        assert cache.get_snapshot_path("owner/model", "v2") is None

    def test_matching_ref_is_returned(self, tmp_path: Path) -> None:
        cache = CacheManager(forgeai_home=tmp_path)
        repo = cache.get_repo_dir("owner/model")
        (repo / "snapshots" / "abc123").mkdir(parents=True)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text("abc123", encoding="utf-8")
        assert cache.get_snapshot_path("owner/model", "main") == repo / "snapshots" / "abc123"


class TestMaxTokensZero:
    def test_zero_is_not_replaced_by_default(self) -> None:
        backend = VLLMBackend(DevToolSettings(model_name="dummy"))
        backend._is_running = True
        backend._streaming_enabled = False
        seen: list[int] = []

        def fake_generate(prompt, max_tokens, *args):  # type: ignore[no-untyped-def]
            seen.append(max_tokens)
            return MagicMock()

        with patch.object(backend, "_generate_vllm", side_effect=fake_generate):
            asyncio.run(backend.generate("hi", max_tokens=0))
            asyncio.run(backend.generate("hi", max_tokens=None))
        assert seen == [0, 512]


class TestFlatQualityProvenance:
    def test_flat_profile_provenance_is_honoured(self, tmp_path: Path) -> None:
        evidence = {
            "auto": {
                "perplexity": 5.0,
                "task_accuracy": 0.5,
                "long_context_accuracy": 0.4,
                "accuracy_scale": "fraction",
                "dataset": "wikitext",
                "evaluator": "lm-eval",
            }
        }
        path = tmp_path / "q.json"
        path.write_text(json.dumps(evidence), encoding="utf-8")
        result = load_quality_evidence(path)
        assert result["auto"]["task_accuracy"] == pytest.approx(50.0)
