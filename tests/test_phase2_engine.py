"""Phase 2 engine-correctness tests (fake async backends, no GPU/vLLM)."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
import threading
import time
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from forgeai.api.routes.ollama import _ndjson_stream_response
from forgeai.api.runtime import SharedRuntimeAdapter
from forgeai.api.streaming import guarded_stream_response, sse_error_frames
from forgeai.core.backends.base import BaseBackend, GenerationResult, StreamStats
from forgeai.core.backends.vllm_backend import VLLMBackend
from forgeai.core.config import DevToolSettings
from forgeai.core.engine import (
    AdmissionVRAMError,
    DevToolEngine,
    EngineKey,
    EngineManager,
    EngineState,
)
from forgeai.utils.gpu import GPUInfo, GPUTopology


class FakeEngine:
    def __init__(self, name: str = "e") -> None:
        self.name = name
        self.is_closed = False
        self.release_gate: asyncio.Event | None = None
        self.shutdown_started = asyncio.Event()

    async def shutdown(self) -> None:
        self.shutdown_started.set()
        if self.release_gate is not None:
            await self.release_gate.wait()
        self.is_closed = True


class FakeBackend(BaseBackend):
    """Async backend that records overlap between concurrent generate calls."""

    def __init__(self, concurrent: bool) -> None:
        super().__init__(DevToolSettings(model_name="m"))
        self._concurrent = concurrent
        self._is_running = True
        self.active = 0
        self.max_active = 0
        self.aborted: list[str] = []

    @property
    def supports_concurrency(self) -> bool:
        return self._concurrent

    @property
    def supports_streaming(self) -> bool:
        return True

    def initialize(self) -> None:  # pragma: no cover
        pass

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        return "p"

    def shutdown(self) -> None:
        pass

    async def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)

    async def generate(self, prompt: str, **kwargs: Any) -> GenerationResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.05)
        finally:
            self.active -= 1
        return GenerationResult(text="x", prompt_tokens=3, completion_tokens=2)

    async def generate_stream(self, prompt: str, **kwargs: Any):  # type: ignore[override]
        yield "a"
        yield "b"

    async def stream_with_stats(self, prompt: str, *, stats: StreamStats, **kwargs: Any):  # type: ignore[override]
        stats.prompt_tokens = 7
        yield "a"
        stats.completion_tokens = 5
        stats.finish_reason = "length"
        yield "b"


def _engine_with(backend: FakeBackend) -> DevToolEngine:
    engine = DevToolEngine(DevToolSettings(model_name="m"))
    engine._backend = backend
    return engine


async def _drive(response: Any, *, cancel_after_first: bool = False, fail_start: bool = False) -> list[str]:
    """Run a Starlette response against a fake ASGI server, returning body chunks."""
    sent: list[dict[str, Any]] = []
    first_chunk = asyncio.Event()
    never = asyncio.Event()

    async def receive() -> dict[str, Any]:
        await never.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if fail_start and message["type"] == "http.response.start":
            raise OSError("client gone")
        sent.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    task = asyncio.ensure_future(response({"type": "http", "asgi": {}}, receive, send))
    if cancel_after_first:
        await asyncio.wait_for(first_chunk.wait(), 2)
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, OSError):
        await task
    await asyncio.sleep(0.05)  # let shielded cleanup finish
    return [m["body"].decode() for m in sent if m["type"] == "http.response.body" and m.get("body")]


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_backend_overlaps(self) -> None:
        backend = FakeBackend(concurrent=True)
        engine = _engine_with(backend)

        async def run() -> None:
            await asyncio.gather(engine.generate("a"), engine.generate("b"))

        asyncio.run(run())
        self.assertEqual(backend.max_active, 2)

    def test_non_concurrent_backend_is_serialized_and_elapsed_excludes_wait(self) -> None:
        backend = FakeBackend(concurrent=False)
        engine = _engine_with(backend)

        async def run() -> list[GenerationResult]:
            return list(await asyncio.gather(engine.generate("a"), engine.generate("b")))

        results = asyncio.run(run())
        self.assertEqual(backend.max_active, 1)
        for r in results:
            self.assertLess(r.elapsed_seconds, 0.09)  # ~0.05, not ~0.10

    def test_stream_stats_are_per_call(self) -> None:
        backend = FakeBackend(concurrent=True)
        engine = _engine_with(backend)

        async def run() -> StreamStats:
            stats = StreamStats()
            chunks = [c async for c in engine.stream_with_stats("p", stats=stats)]
            self.assertEqual(chunks, ["a", "b"])
            return stats

        stats = asyncio.run(run())
        self.assertEqual((stats.prompt_tokens, stats.completion_tokens), (7, 5))
        self.assertEqual(stats.finish_reason, "length")
        self.assertFalse(engine._lock.locked())

    def test_stream_does_not_hold_lock_across_yield(self) -> None:
        backend = FakeBackend(concurrent=False)
        engine = _engine_with(backend)

        async def run() -> None:
            gen = engine.stream_with_stats("p", stats=StreamStats())
            await gen.__anext__()
            self.assertFalse(engine._lock.locked())
            await gen.aclose()

        asyncio.run(run())


class StreamGuardTests(unittest.TestCase):
    def _make(self, inner: Any, record: dict[str, Any], on_error: Any = None) -> Any:
        async def release() -> None:
            record["released"] = record.get("released", 0) + 1

        async def abort() -> None:
            record["aborted"] = True

        async def default_on_error(err: Exception) -> list[str]:
            return sse_error_frames(err)

        return guarded_stream_response(
            inner,
            release=release,
            abort=abort,
            on_error=on_error or default_on_error,
            media_type="text/event-stream",
        )

    def test_cancel_releases_aborts_and_closes_generator(self) -> None:
        record: dict[str, Any] = {}

        async def inner():
            try:
                yield "data: 1\n\n"
                await asyncio.sleep(30)
                yield "data: 2\n\n"
            finally:
                record["closed"] = True

        async def run() -> None:
            await _drive(self._make(inner(), record), cancel_after_first=True)

        asyncio.run(run())
        self.assertEqual(record.get("released"), 1)
        self.assertTrue(record.get("aborted"))
        self.assertTrue(record.get("closed"))

    def test_failed_response_start_still_releases(self) -> None:
        record: dict[str, Any] = {}

        async def inner():
            yield "x"

        async def run() -> None:
            await _drive(self._make(inner(), record), fail_start=True)

        asyncio.run(run())
        self.assertEqual(record.get("released"), 1)

    def test_normal_completion_releases_once_without_abort(self) -> None:
        record: dict[str, Any] = {}

        async def inner():
            yield "x"

        body = asyncio.run(_drive(self._make(inner(), record)))
        self.assertEqual(body, ["x"])
        self.assertEqual(record.get("released"), 1)
        self.assertNotIn("aborted", record)

    def test_midstream_error_emits_sse_error_then_done(self) -> None:
        record: dict[str, Any] = {}

        async def inner():
            yield "data: ok\n\n"
            raise RuntimeError("boom")

        body = asyncio.run(_drive(self._make(inner(), record)))
        self.assertEqual(body[0], "data: ok\n\n")
        self.assertEqual(json.loads(body[1].removeprefix("data: "))["error"]["message"], "boom")
        self.assertEqual(body[-1], "data: [DONE]\n\n")
        self.assertEqual(record.get("released"), 1)

    def test_ndjson_error_line_and_abort_on_cancel(self) -> None:
        released: list[int] = []
        aborted: list[str] = []

        class Runtime:
            async def release_lease(self, key: Any, keep_alive: Any = None) -> None:
                released.append(1)

        class ErrEngine:
            async def generate_stream(self, prompt, **kw):
                yield "hi"
                raise RuntimeError("bad")

        class HangEngine:
            async def generate_stream(self, prompt, **kw):
                yield "hi"
                await asyncio.sleep(30)

            async def abort(self, request_id: str) -> None:
                aborted.append(request_id)

        def make(engine: Any) -> Any:
            return _ndjson_stream_response(
                Runtime(),
                "k",
                engine,
                "p",
                keep_alive=None,
                params={"max_tokens": 1},
                make_chunk=lambda t: {"response": t, "done": False},
                make_terminal=lambda s, a, b: {"done": True, "eval_count": s.completion_tokens},
                acquire_start=time.perf_counter(),
            )

        body = asyncio.run(_drive(make(ErrEngine())))
        self.assertEqual(json.loads(body[0])["response"], "hi")
        self.assertIn("error", json.loads(body[1]))
        self.assertEqual(released, [1])

        asyncio.run(_drive(make(HangEngine()), cancel_after_first=True))
        self.assertEqual(released, [1, 1])
        self.assertEqual(len(aborted), 1)


class ManagerLifecycleTests(unittest.TestCase):
    def _manager(self, engines: dict[str, FakeEngine], **kw: Any) -> EngineManager:
        def factory(key: EngineKey) -> FakeEngine:
            eng = FakeEngine(key.repo_id)
            engines[key.repo_id] = eng
            return eng

        return EngineManager(
            engine_factory=factory,
            max_loaded_models=kw.pop("max_loaded_models", 3),
            load_concurrency=kw.pop("load_concurrency", 1),
            **kw,
        )

    def test_unload_one_keeps_other_and_no_ray_shutdown(self) -> None:
        engines: dict[str, FakeEngine] = {}
        fake_ray = MagicMock()
        fake_ray.is_initialized.return_value = True

        async def run() -> None:
            mgr = self._manager(engines)
            k1, k2 = EngineKey(repo_id="a"), EngineKey(repo_id="b")
            await mgr.acquire(k1, keep_alive=-1)
            await mgr.acquire(k2, keep_alive=-1)
            await mgr.release(k1)
            await mgr.stop(k1)
            self.assertTrue(engines["a"].is_closed)
            self.assertFalse(engines["b"].is_closed)
            fake_ray.shutdown.assert_not_called()
            states = {s.key.repo_id: s.state for s in await mgr.list_async()}
            self.assertEqual(states, {"b": EngineState.READY})
            await mgr.release(k2)
            await mgr.shutdown()
            self.assertTrue(engines["b"].is_closed)
            fake_ray.shutdown.assert_called_once()

        with patch.dict(sys.modules, {"ray": fake_ray}):
            asyncio.run(run())

    def test_stop_with_active_lease_does_not_unload_under_it(self) -> None:
        engines: dict[str, FakeEngine] = {}

        async def run() -> None:
            mgr = self._manager(engines)
            key = EngineKey(repo_id="a")
            await mgr.acquire(key, keep_alive=-1)
            await mgr.stop(key, drain_timeout=0.05)
            self.assertFalse(engines["a"].is_closed)
            st = (await mgr.list_async())[0]
            self.assertEqual(st.state, EngineState.DRAINING)
            await mgr.release(key)
            await mgr.shutdown()  # awaits any in-progress cleanup
            self.assertTrue(engines["a"].is_closed)
            self.assertEqual(await mgr.list_async(), [])

        asyncio.run(run())

    def test_release_of_last_lease_unloads_after_stop_timeout(self) -> None:
        engines: dict[str, FakeEngine] = {}

        async def run() -> None:
            mgr = self._manager(engines)
            key = EngineKey(repo_id="a")
            await mgr.acquire(key, keep_alive=-1)
            await mgr.stop(key, drain_timeout=0.01)
            await mgr.release(key)
            self.assertTrue(engines["a"].is_closed)
            self.assertEqual(await mgr.list_async(), [])

        asyncio.run(run())

    def test_cancelled_stop_mid_cleanup_does_not_leak(self) -> None:
        engines: dict[str, FakeEngine] = {}

        async def run() -> None:
            mgr = self._manager(engines)
            key = EngineKey(repo_id="a")
            await mgr.acquire(key, keep_alive=-1)
            await mgr.release(key)
            engines["a"].release_gate = asyncio.Event()
            stop_task = asyncio.ensure_future(mgr.stop(key))
            await asyncio.wait_for(engines["a"].shutdown_started.wait(), 2)
            stop_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await stop_task
            # Cleanup continues in the background; shutdown must wait for it, not skip it.
            shutdown_task = asyncio.ensure_future(mgr.shutdown())
            await asyncio.sleep(0.05)
            self.assertFalse(shutdown_task.done())
            engines["a"].release_gate.set()
            await asyncio.wait_for(shutdown_task, 2)
            self.assertTrue(engines["a"].is_closed)
            self.assertEqual(await mgr.list_async(), [])

        asyncio.run(run())

    def test_admission_evicts_multiple_idle_engines(self) -> None:
        engines: dict[str, FakeEngine] = {}
        holder: dict[str, EngineManager] = {}

        def estimator(key: EngineKey) -> bool:
            # Fits only when no other engine is resident.
            return all(k == key for k in holder["m"]._entries)

        async def run() -> None:
            mgr = self._manager(engines, admission_estimator=estimator)
            holder["m"] = mgr
            for name in ("a", "b", "c"):
                await mgr.acquire(EngineKey(repo_id=name), keep_alive=-1)
                await mgr.release(EngineKey(repo_id=name))
            await mgr.acquire(EngineKey(repo_id="d"), keep_alive=-1)
            self.assertTrue(all(engines[n].is_closed for n in "abc"))
            self.assertEqual([s.key.repo_id for s in await mgr.list_async()], ["d"])

        asyncio.run(run())

    def test_admission_raises_when_nothing_left_to_evict(self) -> None:
        engines: dict[str, FakeEngine] = {}

        async def run() -> None:
            mgr = self._manager(engines, admission_estimator=lambda k: False)
            with self.assertRaises(AdmissionVRAMError):
                await mgr.acquire(EngineKey(repo_id="a"))

        asyncio.run(run())

    def test_finished_ttl_timers_are_forgotten(self) -> None:
        engines: dict[str, FakeEngine] = {}

        async def run() -> None:
            mgr = self._manager(engines)
            key = EngineKey(repo_id="a")
            await mgr.acquire(key, keep_alive="50ms")
            await mgr.release(key)
            self.assertIn(key, mgr._timer_tasks)
            await asyncio.sleep(0.2)
            self.assertNotIn(key, mgr._timer_tasks)
            self.assertTrue(engines["a"].is_closed)

        asyncio.run(run())

    def test_shutdown_does_not_wait_on_caller_tasks(self) -> None:
        engines: dict[str, FakeEngine] = {}

        async def run() -> None:
            mgr = self._manager(engines)
            await mgr.acquire(EngineKey(repo_id="a"), keep_alive=-1)
            self.assertEqual(mgr._in_progress_loads, set())

        asyncio.run(run())


class RuntimeAdapterTests(unittest.TestCase):
    def _manifest(self) -> SimpleNamespace:
        return SimpleNamespace(
            model="org/model",
            revision="rev1",
            chat_template="",
            tokenizer_override="",
            engine_settings=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                weight_quantization="none",
                gpu_memory_utilization=0.8,
                enforce_eager=False,
                trust_remote_code=False,
            ),
            kv_cache=SimpleNamespace(dtype="auto"),
        )

    def _record(self, path: str, size: int) -> SimpleNamespace:
        return SimpleNamespace(
            snapshot_path=path,
            size_bytes=size,
            ref=SimpleNamespace(digest="d", full_tag="m:latest"),
        )

    def _adapter(self, mgr: EngineManager | None = None) -> SharedRuntimeAdapter:
        return SharedRuntimeAdapter(
            engine_manager=mgr or EngineManager(engine_factory=lambda k: FakeEngine()),
            model_registry=MagicMock(),
            cache_manager=MagicMock(),
        )

    def test_engine_key_is_stable_across_download_state(self) -> None:
        adapter = self._adapter()
        manifest = self._manifest()
        k_missing = adapter.build_engine_key(manifest, self._record("/hub/models--org--model", 0))
        k_present = adapter.build_engine_key(manifest, self._record("/hub/snap/abc", 12345))
        self.assertEqual(k_missing, k_present)

    def test_key_to_tag_is_pruned_on_unload(self) -> None:
        async def run() -> None:
            adapter = self._adapter()
            manifest, record = self._manifest(), self._record("/p", 1)
            key = adapter.build_engine_key(manifest, record)
            adapter.get_engine_key_for_tag = lambda tag: (key, manifest, record)  # type: ignore[method-assign]
            await adapter.acquire_lease("m:latest", keep_alive=0)
            self.assertEqual(adapter.get_public_tag_for_key(key), "m:latest")
            await adapter.release_lease(key)
            self.assertNotIn(key, adapter._key_to_tag)

        asyncio.run(run())


class BackendTests(unittest.TestCase):
    def test_startup_context_lock_serializes_quiet_loads(self) -> None:
        state = {"active": 0, "max": 0}
        guard = threading.Lock()

        def load() -> None:
            backend = VLLMBackend(DevToolSettings(model_name="m"), quiet_startup=True)
            with backend._startup_context():
                with guard:
                    state["active"] += 1
                    state["max"] = max(state["max"], state["active"])
                time.sleep(0.05)
                with guard:
                    state["active"] -= 1

        threads = [threading.Thread(target=load) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(state["max"], 1)

    def test_preflight_requires_tp_times_pp_gpus(self) -> None:
        settings = DevToolSettings(
            model_name="m", tensor_parallel_size=1, pipeline_parallel_size=2
        )
        topology = GPUTopology(
            gpus=[GPUInfo(index=0, name="g", total_memory_mb=8192, free_memory_mb=8000)]
        )
        with self.assertRaises(RuntimeError) as ctx:
            VLLMBackend(settings)._preflight_vllm_memory(topology)
        self.assertIn("requires 2 GPUs", str(ctx.exception))

    def test_initialize_detects_gpus_once(self) -> None:
        topology = GPUTopology(
            gpus=[GPUInfo(index=0, name="g", total_memory_mb=8192, free_memory_mb=8000)]
        )
        backend = VLLMBackend(DevToolSettings(model_name="m", enforce_version_check=False))
        with (
            patch("forgeai.utils.gpu.detect_gpus", return_value=topology) as detect,
            patch.object(VLLMBackend, "_init_vllm"),
        ):
            backend.initialize()
        self.assertEqual(detect.call_count, 1)

    def test_vllm_stream_stats_and_abort(self) -> None:
        backend = VLLMBackend(DevToolSettings(model_name="m"), streaming=True)
        backend._is_running = True
        aborted: list[str] = []

        class FakeAsyncEngine:
            async def generate(self, prompt, params, request_id):
                self.request_id = request_id
                yield SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[SimpleNamespace(text="Hel", token_ids=[10, 11], finish_reason=None)],
                )
                yield SimpleNamespace(
                    prompt_token_ids=[1, 2, 3],
                    outputs=[SimpleNamespace(text="lo", token_ids=[12], finish_reason="length")],
                )

            async def abort(self, request_id):
                aborted.append(request_id)

        backend._engine = FakeAsyncEngine()
        fake_vllm = ModuleType("vllm")
        fake_vllm.SamplingParams = lambda **kw: SimpleNamespace(**kw)  # type: ignore[attr-defined]
        fake_sp = ModuleType("vllm.sampling_params")
        fake_sp.RequestOutputKind = SimpleNamespace(DELTA="delta")  # type: ignore[attr-defined]

        async def run() -> StreamStats:
            stats = StreamStats(request_id="rid-1")
            chunks = [c async for c in backend.stream_with_stats("p", stats=stats, max_tokens=4)]
            self.assertEqual(chunks, ["Hel", "lo"])
            await backend.abort("rid-1")
            return stats

        with patch.dict(sys.modules, {"vllm": fake_vllm, "vllm.sampling_params": fake_sp}):
            stats = asyncio.run(run())
        self.assertEqual(
            (stats.prompt_tokens, stats.completion_tokens, stats.finish_reason), (3, 3, "length")
        )
        self.assertEqual(backend._engine.request_id, "rid-1")
        self.assertEqual(aborted, ["rid-1"])


if __name__ == "__main__":
    unittest.main()
