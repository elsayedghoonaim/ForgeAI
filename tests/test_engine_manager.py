from __future__ import annotations

import asyncio
import sys
import threading
import unittest
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.core.backends.vllm_backend import validate_turboquant_hardware
from forgeai.core.config import DevToolSettings, KVCacheSettings, QuantizationType
from forgeai.core.engine import (
    AdmissionVRAMError,
    DevToolEngine,
    EngineDrainingError,
    EngineKey,
    EngineLoadCancelledError,
    EngineManager,
    EngineQueueFullError,
    EngineShuttingDownError,
    NoCompatibleGPUError,
    ROCmDeferredError,
    TurboQuantHWCCError,
    parse_keep_alive,
)
from forgeai.utils.gpu import GPUInfo, GPUTopology


class FakeAsyncEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.is_closed = False
        self.shutdown_thread_id: int | None = None

    async def shutdown(self) -> None:
        self.shutdown_thread_id = threading.get_ident()
        self.is_closed = True


class FakeSyncEngine:
    def __init__(self, name: str) -> None:
        self.name = name
        self.is_closed = False
        self.shutdown_thread_id: int | None = None

    def shutdown(self) -> None:
        self.shutdown_thread_id = threading.get_ident()
        self.is_closed = True


class TestKeepAliveParser(unittest.TestCase):
    def test_parse_keep_alive_defaults_and_numbers(self) -> None:
        p1 = parse_keep_alive(None)
        self.assertEqual(p1.ttl_seconds, 300.0)

        p2 = parse_keep_alive(60)
        self.assertEqual(p2.ttl_seconds, 60.0)
        self.assertFalse(p2.is_indefinite)
        self.assertFalse(p2.is_immediate_unload)

        p3 = parse_keep_alive(0)
        self.assertEqual(p3.ttl_seconds, 0.0)
        self.assertTrue(p3.is_immediate_unload)

        p4 = parse_keep_alive(-1)
        self.assertEqual(p4.ttl_seconds, -1.0)
        self.assertTrue(p4.is_indefinite)

    def test_parse_keep_alive_duration_strings_and_unicode_micro_sign(self) -> None:
        self.assertEqual(parse_keep_alive("5m").ttl_seconds, 300.0)
        self.assertEqual(parse_keep_alive("10s").ttl_seconds, 10.0)
        self.assertEqual(parse_keep_alive("1h").ttl_seconds, 3600.0)
        self.assertEqual(parse_keep_alive("500ms").ttl_seconds, 0.5)
        self.assertEqual(parse_keep_alive("1000us").ttl_seconds, 0.001)
        # Using literal actual Unicode micro sign µ (U+00B5) and Greek mu μ (U+03BC)
        self.assertEqual(parse_keep_alive("1000µs").ttl_seconds, 0.001)
        self.assertEqual(parse_keep_alive("1000μs").ttl_seconds, 0.001)
        self.assertEqual(parse_keep_alive("1000000ns").ttl_seconds, 0.001)
        self.assertTrue(parse_keep_alive("0s").is_immediate_unload)
        self.assertTrue(parse_keep_alive("-5m").is_indefinite)

    def test_parse_keep_alive_rejections(self) -> None:
        with self.assertRaises(ValueError):
            parse_keep_alive(True)
        with self.assertRaises(ValueError):
            parse_keep_alive(False)
        with self.assertRaises(ValueError):
            parse_keep_alive(float("nan"))
        with self.assertRaises(ValueError):
            parse_keep_alive(float("inf"))
        with self.assertRaises(ValueError):
            parse_keep_alive("invalid_string")
        with self.assertRaises(ValueError):
            parse_keep_alive("500xx")
        with self.assertRaises(ValueError):
            parse_keep_alive(9999999999)


class TestEngineKeyAndConfig(unittest.TestCase):
    def test_engine_key_identity_propagation_and_validation(self) -> None:
        key = EngineKey(
            repo_id="model/a",
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            dtype="bfloat16",
            weight_quantization="awq",
            kv_cache_dtype="turboquant_4bit_nc",
            max_model_len=8192,
        )
        settings = key.to_settings()
        kwargs = settings.to_vllm_kwargs()

        self.assertEqual(settings.model_name, "model/a")
        self.assertEqual(settings.tensor_parallel_size, 2)
        self.assertEqual(settings.pipeline_parallel_size, 4)
        self.assertEqual(settings.dtype, "bfloat16")
        self.assertEqual(settings.kv_cache_dtype, "turboquant_4bit_nc")
        self.assertEqual(settings.max_model_len, 8192)

        self.assertEqual(kwargs["quantization"], "awq")
        self.assertEqual(kwargs["kv_cache_dtype"], "turboquant_4bit_nc")
        self.assertEqual(kwargs["pipeline_parallel_size"], 4)
        self.assertEqual(kwargs["dtype"], "bfloat16")

        with self.assertRaises(ValueError):
            EngineKey(repo_id="model/invalid", weight_quantization="unsupported_quant")

    def test_quantization_type_extension(self) -> None:
        self.assertEqual(QuantizationType.FP8.value, "fp8")
        self.assertEqual(QuantizationType.BITSANDBYTES.value, "bitsandbytes")

    def test_config_kv_cache_dtype_validation(self) -> None:
        settings = DevToolSettings(kv_cache_dtype="turboquant_4bit_nc")
        self.assertEqual(settings.kv_cache_dtype, "turboquant_4bit_nc")

        with self.assertRaises(ValueError):
            DevToolSettings(kv_cache_dtype="turboquant_3bit_nc")

        with self.assertRaises(ValueError):
            DevToolSettings(kv_cache_dtype="unsupported_format")

        kv_set = KVCacheSettings(dtype="fp8")
        self.assertEqual(kv_set.dtype, "fp8")

    def test_load_concurrency_exceeding_max_loaded_models_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DevToolSettings(max_loaded_models=1, load_concurrency=2)

        with self.assertRaises(ValueError):
            EngineManager(max_loaded_models=1, load_concurrency=2)


class TestTurboQuantHardwareGates(unittest.TestCase):
    def test_rocm_deferred_error(self) -> None:
        with self.assertRaises(ROCmDeferredError) as ctx:
            validate_turboquant_hardware("turboquant_4bit_nc", is_rocm=True)
        self.assertEqual(ctx.exception.code, "ERR_ROCM_DEFERRED")

    def test_no_gpu_error(self) -> None:
        topo = GPUTopology(gpus=[])
        with self.assertRaises(NoCompatibleGPUError) as ctx:
            validate_turboquant_hardware(
                "turboquant_k8v4", gpu_detector=lambda: topo, is_rocm=False
            )
        self.assertEqual(ctx.exception.code, "ERR_NO_COMPATIBLE_GPU")

    def test_low_cc_and_zero_cc_error(self) -> None:
        topo = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="NVIDIA GTX 1080",
                    compute_capability=(6, 1),
                )
            ]
        )
        with self.assertRaises(TurboQuantHWCCError) as ctx:
            validate_turboquant_hardware(
                "turboquant_4bit_nc", gpu_detector=lambda: topo, is_rocm=False
            )
        self.assertEqual(ctx.exception.code, "ERR_TURBOQUANT_HW_CC")

        topo_zero = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="Unknown GPU",
                    compute_capability=(0, 0),
                )
            ]
        )
        with self.assertRaises(TurboQuantHWCCError) as ctx:
            validate_turboquant_hardware(
                "turboquant_4bit_nc", gpu_detector=lambda: topo_zero, is_rocm=False
            )
        self.assertEqual(ctx.exception.code, "ERR_TURBOQUANT_HW_CC")

    def test_valid_hardware_passes(self) -> None:
        topo = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="NVIDIA RTX 4090",
                    compute_capability=(8, 9),
                )
            ]
        )
        validate_turboquant_hardware(
            "turboquant_4bit_nc", gpu_detector=lambda: topo, is_rocm=False
        )
        validate_turboquant_hardware("auto", gpu_detector=lambda: topo, is_rocm=False)
        validate_turboquant_hardware("fp8", gpu_detector=lambda: topo, is_rocm=False)


class TestEngineManagerLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.factory_call_count = 0
        self.cleaned_up_engines: list[Any] = []

    async def async_factory(self, key: EngineKey) -> FakeAsyncEngine:
        self.factory_call_count += 1
        return FakeAsyncEngine(key.repo_id)

    async def async_cleanup_hook(self, engine: Any) -> None:
        self.cleaned_up_engines.append(engine)

    def test_max_concurrency_serialization_with_multiple_models(self) -> None:
        async def run() -> None:
            factory_release = asyncio.Event()
            current_factory_executions = 0
            peak_factory_executions = 0

            async def concurrency_factory(key: EngineKey) -> FakeAsyncEngine:
                nonlocal current_factory_executions, peak_factory_executions
                current_factory_executions += 1
                if current_factory_executions > peak_factory_executions:
                    peak_factory_executions = current_factory_executions
                await factory_release.wait()
                current_factory_executions -= 1
                return FakeAsyncEngine(key.repo_id)

            manager = EngineManager(
                engine_factory=concurrency_factory,
                cleanup_hook=self.async_cleanup_hook,
                max_loaded_models=2,
                load_concurrency=1,
            )
            key1 = EngineKey(repo_id="model/c1")
            key2 = EngineKey(repo_id="model/c2")

            t1 = asyncio.create_task(manager.acquire(key1))
            t2 = asyncio.create_task(manager.acquire(key2))

            await asyncio.sleep(0.05)

            factory_release.set()
            lease1, lease2 = await asyncio.gather(t1, t2)

            self.assertEqual(peak_factory_executions, 1)
            self.assertEqual(lease1.key.repo_id, "model/c1")
            self.assertEqual(lease2.key.repo_id, "model/c2")

            await manager.release(key1)
            await manager.release(key2)
            await manager.shutdown()

        asyncio.run(run())

    def test_sync_blocking_factory_cancellation_worker_cleanup(self) -> None:
        async def run() -> None:
            factory_started = threading.Event()
            factory_release = threading.Event()
            constructed_engine: FakeSyncEngine | None = None

            def blocking_sync_factory(key: EngineKey) -> FakeSyncEngine:
                nonlocal constructed_engine
                factory_started.set()
                factory_release.wait()
                constructed_engine = FakeSyncEngine(key.repo_id)
                return constructed_engine

            manager = EngineManager(
                engine_factory=blocking_sync_factory,
                cleanup_hook=self.async_cleanup_hook,
            )
            key = EngineKey(repo_id="model/sync_cancel")

            t_leader = asyncio.create_task(manager.acquire(key))
            await asyncio.to_thread(factory_started.wait)

            t_follower = asyncio.create_task(manager.acquire(key))
            await asyncio.sleep(0.02)

            t_leader.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t_leader

            with self.assertRaises(EngineLoadCancelledError) as ctx:
                await t_follower
            self.assertEqual(ctx.exception.code, "ERR_ENGINE_LOAD_CANCELLED")

            self.assertEqual(len(await manager.list_async()), 0)

            factory_release.set()
            await manager.shutdown()

            assert constructed_engine is not None
            self.assertTrue(constructed_engine.is_closed)
            self.assertIn(constructed_engine, self.cleaned_up_engines)

        asyncio.run(run())

    def test_sync_shutdown_and_sync_hook_run_off_loop_thread(self) -> None:
        async def run() -> None:
            main_thread_id = threading.get_ident()
            sync_engine = FakeSyncEngine("model/sync")
            hook_thread_id: int | None = None

            def sync_hook(engine: Any) -> None:
                nonlocal hook_thread_id
                hook_thread_id = threading.get_ident()
                self.cleaned_up_engines.append(engine)

            async def sync_factory(key: EngineKey) -> FakeSyncEngine:
                return sync_engine

            manager = EngineManager(
                engine_factory=sync_factory,
                cleanup_hook=sync_hook,
            )
            key = EngineKey(repo_id="model/sync")

            await manager.acquire(key)
            await manager.release(key, keep_alive=0)
            await asyncio.sleep(0.05)

            self.assertTrue(sync_engine.is_closed)
            self.assertIsNotNone(sync_engine.shutdown_thread_id)
            self.assertNotEqual(sync_engine.shutdown_thread_id, main_thread_id)

            self.assertIsNotNone(hook_thread_id)
            self.assertNotEqual(hook_thread_id, main_thread_id)

            await manager.shutdown()

        asyncio.run(run())

    def test_ordered_different_key_capacity_race_fairness(self) -> None:
        async def run() -> None:
            factory_entered_1 = asyncio.Event()
            factory_release_1 = asyncio.Event()

            async def ordered_factory(key: EngineKey) -> FakeAsyncEngine:
                if key.repo_id == "model/1":
                    factory_entered_1.set()
                    await factory_release_1.wait()
                return FakeAsyncEngine(key.repo_id)

            manager = EngineManager(
                engine_factory=ordered_factory,
                cleanup_hook=self.async_cleanup_hook,
                max_loaded_models=1,
                load_concurrency=1,
            )

            key1 = EngineKey(repo_id="model/1")
            key2 = EngineKey(repo_id="model/2")

            t1 = asyncio.create_task(manager.acquire(key1))
            await factory_entered_1.wait()

            with self.assertRaises(AdmissionVRAMError) as ctx:
                await manager.acquire(key2)
            self.assertEqual(ctx.exception.code, "ERR_ADMISSION_VRAM")

            factory_release_1.set()
            lease1 = await t1
            self.assertEqual(lease1.key.repo_id, "model/1")

            await manager.release(key1)
            await manager.shutdown()

        asyncio.run(run())

    def test_follower_keep_alive_policy_applied(self) -> None:
        async def run() -> None:
            factory_entered = asyncio.Event()
            factory_release = asyncio.Event()

            async def slow_factory(key: EngineKey) -> FakeAsyncEngine:
                factory_entered.set()
                await factory_release.wait()
                return FakeAsyncEngine(key.repo_id)

            manager = EngineManager(
                engine_factory=slow_factory,
                cleanup_hook=self.async_cleanup_hook,
                default_keep_alive="5m",
            )
            key = EngineKey(repo_id="model/ka")

            t_leader = asyncio.create_task(manager.acquire(key, keep_alive="5m"))
            await factory_entered.wait()

            t_follower = asyncio.create_task(manager.acquire(key, keep_alive="10s"))

            factory_release.set()
            lease_leader, lease_follower = await asyncio.gather(t_leader, t_follower)

            self.assertEqual(lease_follower.keep_alive_policy.ttl_seconds, 10.0)

            await manager.release(key)
            await manager.release(key)
            await manager.shutdown()

        asyncio.run(run())

    def test_follower_cancellation_isolation(self) -> None:
        async def run() -> None:
            factory_entered = asyncio.Event()
            factory_release = asyncio.Event()

            async def slow_factory(key: EngineKey) -> FakeAsyncEngine:
                factory_entered.set()
                await factory_release.wait()
                return FakeAsyncEngine(key.repo_id)

            manager = EngineManager(
                engine_factory=slow_factory,
                cleanup_hook=self.async_cleanup_hook,
            )
            key = EngineKey(repo_id="model/cancel_follower")

            t_leader = asyncio.create_task(manager.acquire(key))
            await factory_entered.wait()

            t_follower = asyncio.create_task(manager.acquire(key))
            await asyncio.sleep(0.02)

            t_follower.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await t_follower

            factory_release.set()
            lease_leader = await t_leader
            self.assertEqual(lease_leader.key.repo_id, "model/cancel_follower")

            await manager.release(key)
            await manager.shutdown()

        asyncio.run(run())

    def test_lock_probe_cleanup_hook_proves_metadata_lock_not_held(self) -> None:
        async def run() -> None:
            lock_acquired_during_cleanup = False

            async def lock_probing_cleanup(engine: Any) -> None:
                nonlocal lock_acquired_during_cleanup
                statuses = await manager.list_async()
                if isinstance(statuses, list):
                    lock_acquired_during_cleanup = True
                self.cleaned_up_engines.append(engine)

            manager = EngineManager(
                engine_factory=self.async_factory,
                cleanup_hook=lock_probing_cleanup,
            )
            key = EngineKey(repo_id="model/lock_probe")

            await manager.acquire(key)
            await manager.release(key, keep_alive=0)
            await asyncio.sleep(0.05)

            self.assertTrue(lock_acquired_during_cleanup)
            await manager.shutdown()

        asyncio.run(run())

    def test_draining_reacquire_refused(self) -> None:
        async def run() -> None:
            manager = EngineManager(
                engine_factory=self.async_factory,
                cleanup_hook=self.async_cleanup_hook,
            )
            key = EngineKey(repo_id="model/draining_test")
            await manager.acquire(key)

            stop_task = asyncio.create_task(manager.stop(key, drain_timeout=5.0))
            await asyncio.sleep(0.02)

            with self.assertRaises(EngineDrainingError) as ctx:
                await manager.acquire(key)
            self.assertEqual(ctx.exception.code, "ERR_ENGINE_DRAINING")

            await manager.release(key)
            await stop_task
            await manager.shutdown()

        asyncio.run(run())

    def test_queue_full_refused(self) -> None:
        async def run() -> None:
            factory_entered = asyncio.Event()
            factory_release = asyncio.Event()

            async def slow_factory(key: EngineKey) -> FakeAsyncEngine:
                factory_entered.set()
                await factory_release.wait()
                return FakeAsyncEngine(key.repo_id)

            manager = EngineManager(
                engine_factory=slow_factory,
                cleanup_hook=self.async_cleanup_hook,
                request_queue_depth=1,
            )
            key1 = EngineKey(repo_id="model/q1")
            key2 = EngineKey(repo_id="model/q2")

            t1 = asyncio.create_task(manager.acquire(key1))
            await factory_entered.wait()

            with self.assertRaises(EngineQueueFullError) as ctx:
                await manager.acquire(key2)
            self.assertEqual(ctx.exception.code, "ERR_QUEUE_FULL")

            factory_release.set()
            await t1
            await manager.release(key1)
            await manager.shutdown()

        asyncio.run(run())

    def test_shutdown_refuses_new_acquire(self) -> None:
        async def run() -> None:
            manager = EngineManager(
                engine_factory=self.async_factory,
                cleanup_hook=self.async_cleanup_hook,
            )
            key = EngineKey(repo_id="model/shut")
            await manager.shutdown()

            with self.assertRaises(EngineShuttingDownError) as ctx:
                await manager.acquire(key)
            self.assertEqual(ctx.exception.code, "ERR_ENGINE_SHUTTING_DOWN")

        asyncio.run(run())

    def test_dev_tool_engine_source_compatibility(self) -> None:
        settings = DevToolSettings(model_name="test-model")
        engine = DevToolEngine(settings)
        self.assertFalse(engine.is_running)


if __name__ == "__main__":
    unittest.main()
