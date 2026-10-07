from __future__ import annotations

import pytest

from forgeai.benchmarking.turboquant import ProfileMeasurements, max_allowed_unload_vram_mb
from forgeai.core.config import RUNTIME_KV_CACHE_DTYPES, DevToolSettings, reject_gguf
from forgeai.core.resource_profiles import PRIMARY_PROFILES
from forgeai.core.security import vllm_version_matches
from forgeai.core.telemetry import TelemetryCollector
from forgeai.models.manifest import KVCacheSettings


def test_primary_profiles_are_accepted_by_config() -> None:
    assert PRIMARY_PROFILES == RUNTIME_KV_CACHE_DTYPES
    for dtype in PRIMARY_PROFILES:
        assert DevToolSettings(kv_cache_dtype=dtype).kv_cache_dtype == dtype
        assert KVCacheSettings(dtype=dtype).dtype == dtype
    with pytest.raises(ValueError):
        KVCacheSettings(dtype="turboquant_3bit_nc")


def test_reject_gguf() -> None:
    reject_gguf("org/model")
    reject_gguf(None)
    with pytest.raises(ValueError, match="GGUF model format is unsupported"):
        reject_gguf("a/b.Q4.GGUF")


def test_vllm_version_matches_ignores_local_metadata() -> None:
    assert vllm_version_matches("0.30.0+cu128")
    assert not vllm_version_matches("0.30.1")
    assert not vllm_version_matches("garbage")


def test_leak_threshold() -> None:
    assert max_allowed_unload_vram_mb(0) == 100.0
    assert max_allowed_unload_vram_mb(24000) == 480.0


def test_measurements_from_dict_coerces_and_validates() -> None:
    m = ProfileMeasurements.from_dict(
        {"ttft_p95_ms": "12.5", "crashes_count": 2.0, "has_memory_leak": False, "unknown": 1}
    )
    assert m.ttft_p95_ms == 12.5 and m.crashes_count == 2 and m.has_memory_leak is False
    for bad in ({"ttft_p95_ms": "abc"}, {"ttft_p95_ms": float("nan")}, {"nans_count": True},
                {"crashes_count": 1.5}, {"has_memory_leak": "no"}):
        with pytest.raises(ValueError):
            ProfileMeasurements.from_dict(bad)


def test_telemetry_buffer_bounded_and_flushed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    t = TelemetryCollector(enabled=True, storage_dir=str(tmp_path))
    for i in range(TelemetryCollector.MAX_BUFFERED_EVENTS + 50):
        t.track("e", {"i": i})
    assert len(t._events) == TelemetryCollector.MAX_BUFFERED_EVENTS
    t.shutdown()
    assert not t._events
    assert list(tmp_path.glob("events_*.jsonl"))
