"""Phase 3 performance tests: manifest/snapshot caching, non-blocking handlers,
HF cache accounting, download filtering and pure-ASGI middleware."""

from __future__ import annotations

import asyncio
import os
import pathlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from forgeai.api import server as server_mod
from forgeai.api.runtime import SharedRuntimeAdapter
from forgeai.api.server import RequestLoggingMiddleware, create_app
from forgeai.models import loader as loader_mod
from forgeai.models import registry as registry_mod
from forgeai.models.loader import (
    DOWNLOAD_ALLOW_PATTERNS,
    CacheManager,
    _dir_size_no_follow,
    get_cached_models,
)
from forgeai.models.manifest import ForgeAIManifest
from forgeai.models.registry import ModelRegistry
from forgeai.monitoring.metrics import ACTIVE_REQUESTS, REQUESTS_TOTAL
from forgeai.utils.helpers import format_bytes


def _hf_manifest(name: str = "demo:latest", model: str = "org/demo", revision: str = "main"):
    return ForgeAIManifest(name=name, model=model, revision=revision)


def _make_registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(manifests_dir=tmp_path / "home" / "manifests")


def _fake_snapshot(cache: CacheManager, repo: str, commit: str, revision: str, payload: int) -> Path:
    repo_dir = cache.get_repo_dir(repo)
    blobs = repo_dir / "blobs"
    snap = repo_dir / "snapshots" / commit
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs").mkdir(exist_ok=True)
    (repo_dir / "refs" / revision).write_text(commit, encoding="utf-8")
    blob = blobs / f"blob-{commit}"
    blob.write_bytes(b"x" * payload)
    os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
    return snap


# --------------------------------------------------------------------- C3: caching


def test_manifest_cache_hit_avoids_reparse_and_invalidates_on_mtime(tmp_path: Path) -> None:
    writer = _make_registry(tmp_path)
    writer.register_manifest(_hf_manifest())
    path = writer._get_manifest_path("demo:latest")

    reader = _make_registry(tmp_path)  # cold cache
    calls = 0
    real_from_yaml = ForgeAIManifest.from_yaml.__func__  # type: ignore[attr-defined]

    def counting_from_yaml(cls: Any, text: str) -> Any:
        nonlocal calls
        calls += 1
        return real_from_yaml(cls, text)

    with patch.object(ForgeAIManifest, "from_yaml", classmethod(counting_from_yaml)):
        for _ in range(5):
            record = reader.get_record("demo:latest")
            reader.get_manifest("demo:latest")
            reader.list_records()
        assert calls == 1
        assert record.manifest is not None and record.manifest.model == "org/demo"

        # Rewrite with different content and a new mtime: cache must be invalidated.
        new_yaml = _hf_manifest(model="org/other").to_yaml()
        path.write_text(new_yaml, encoding="utf-8")
        st = path.stat()
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
        assert reader.get_manifest("demo:latest").model == "org/other"
        assert calls == 2


def test_cached_manifest_is_not_shared_mutably(tmp_path: Path) -> None:
    reg = _make_registry(tmp_path)
    reg.register_manifest(_hf_manifest())
    first = reg.get_manifest("demo:latest")
    first.revision = "tampered"
    assert reg.get_manifest("demo:latest").revision == "main"
    rec = reg.get_record("demo:latest")
    assert rec.manifest is not None
    rec.manifest.revision = "tampered"
    assert reg.get_record("demo:latest").manifest.revision == "main"  # type: ignore[union-attr]


def test_register_unregister_invalidate_cache(tmp_path: Path) -> None:
    reg = _make_registry(tmp_path)
    reg.register_manifest(_hf_manifest())
    assert reg.get_manifest("demo:latest").model == "org/demo"
    reg.register_manifest(_hf_manifest(model="org/changed"))
    assert reg.get_manifest("demo:latest").model == "org/changed"
    assert reg.unregister_tag("demo:latest") is not None
    with pytest.raises(KeyError):
        reg.get_manifest("demo:latest")
    with pytest.raises(KeyError):
        reg.get_record("demo:latest")


def test_local_dir_safetensors_validation_is_cached(tmp_path: Path) -> None:
    model_dir = tmp_path / "weights"
    (model_dir / "sub").mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "sub" / "model.safetensors").write_bytes(b"1")

    ForgeAIManifest(name="loc:latest", model=str(model_dir), source_kind="local_dir")

    calls = 0
    real_rglob = pathlib.Path.rglob

    def counting_rglob(self: Path, pattern: str):
        nonlocal calls
        if pattern == "*.safetensors":
            calls += 1
        return real_rglob(self, pattern)

    with patch.object(pathlib.Path, "rglob", counting_rglob):
        for _ in range(4):
            ForgeAIManifest(name="loc:latest", model=str(model_dir), source_kind="local_dir")
    assert calls == 0


def test_snapshot_path_and_size_are_cached_and_lazy(tmp_path: Path) -> None:
    cache = CacheManager(forgeai_home=tmp_path / "home")
    snap = _fake_snapshot(cache, "org/demo", "c1", "main", 1234)
    reg = ModelRegistry(manifests_dir=tmp_path / "home" / "manifests")
    reg.register_manifest(_hf_manifest())

    size_calls = 0
    real_size = registry_mod._dir_size

    def counting_size(p: Path) -> int:
        nonlocal size_calls
        size_calls += 1
        return real_size(p)

    snap_calls = 0
    real_get = cache.get_snapshot_path

    def counting_snap(repo: str, rev: str = "main"):
        nonlocal snap_calls
        snap_calls += 1
        return real_get(repo, rev)

    cache.get_snapshot_path = counting_snap  # type: ignore[method-assign]
    with patch.object(registry_mod, "_dir_size", counting_size):
        for _ in range(5):
            rec = reg.get_record("demo:latest", cache_manager=cache)
        assert rec.snapshot_path == str(snap)
        assert rec.size_bytes == 0 and size_calls == 0  # lazy: hot path never sizes
        assert snap_calls == 1

        for _ in range(3):
            listed = reg.list_records(cache_manager=cache)
        assert listed[0].size_bytes == 1234
        assert size_calls == 1  # computed once, cached by snapshot path

        reg.invalidate()
        assert reg.list_records(cache_manager=cache)[0].size_bytes == 1234
        assert size_calls == 2


# ----------------------------------------------------- C3: handlers stay non-blocking


def _runtime(tmp_path: Path) -> SharedRuntimeAdapter:
    cache = CacheManager(forgeai_home=tmp_path / "home")
    reg = ModelRegistry(manifests_dir=tmp_path / "home" / "manifests")
    reg.register_manifest(_hf_manifest())
    em = MagicMock()
    em.list_async = AsyncMock(return_value=[])
    em.stop = AsyncMock()
    return SharedRuntimeAdapter(engine_manager=em, model_registry=reg, cache_manager=cache)


def test_list_and_delete_endpoints_use_to_thread(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    app = create_app(runtime_adapter=runtime)
    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy(func: Any, *args: Any, **kwargs: Any):
        offloaded.append(getattr(func, "__name__", str(func)))
        return await real_to_thread(func, *args, **kwargs)

    async def run() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            codes = [
                (await client.get("/api/tags")).status_code,
                (await client.get("/v1/models")).status_code,
                (await client.get("/v1/models/demo:latest")).status_code,
                (await client.post("/api/show", json={"model": "demo"})).status_code,
                (await client.request("DELETE", "/api/delete", json={"model": "demo"})).status_code,
            ]
        return codes

    with (
        patch("forgeai.api.routes.ollama.asyncio.to_thread", spy),
        patch("forgeai.api.routes.models.asyncio.to_thread", spy),
        patch("forgeai.api.runtime.asyncio.to_thread", spy),
    ):
        codes = asyncio.run(run())

    assert codes == [200, 200, 200, 200, 200]
    for name in ("list_records", "get_record", "get_manifest_and_record", "get_engine_key_for_tag", "unregister_tag"):
        assert name in offloaded, (name, offloaded)


def test_slow_registry_does_not_block_event_loop(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    app = create_app(runtime_adapter=runtime)
    real = runtime.model_registry.list_records

    def slow(*args: Any, **kwargs: Any):
        time.sleep(0.3)
        return real(*args, **kwargs)

    runtime.model_registry.list_records = slow  # type: ignore[method-assign]

    async def run() -> float:
        max_gap = 0.0

        async def ticker() -> None:
            nonlocal max_gap
            last = time.monotonic()
            for _ in range(25):
                await asyncio.sleep(0.01)
                now = time.monotonic()
                max_gap = max(max_gap, now - last)
                last = now

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            tick = asyncio.create_task(ticker())
            resp = await client.get("/api/tags")
            await tick
            assert resp.status_code == 200
        return max_gap

    assert asyncio.run(run()) < 0.15


# ----------------------------------------------- engine key: local dir vs HF revision


def test_engine_key_local_dir_loads_from_local_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = tmp_path / "local-weights"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"1")
    runtime = _runtime(tmp_path)
    runtime.model_registry.register_manifest(
        ForgeAIManifest(name="loc:latest", model=str(model_dir), source_kind="local_dir")
    )
    monkeypatch.chdir(tmp_path / "home")  # unrelated cwd

    key, manifest, _record = runtime.get_engine_key_for_tag("loc")
    settings = key.to_settings()
    assert manifest.source_kind == "local_dir"
    assert settings.model_path == str(model_dir.resolve())
    # Stable across calls (cache hits) and independent of download state.
    assert runtime.get_engine_key_for_tag("loc")[0] == key


def test_engine_key_hf_manifest_pins_revision_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(tmp_path)
    runtime.model_registry.register_manifest(_hf_manifest(name="pinned:v1", revision="abc123"))
    monkeypatch.chdir(tmp_path)  # no local "org/demo" directory shadowing the repo id

    key_before, _m, record_before = runtime.get_engine_key_for_tag("pinned:v1")
    settings = key_before.to_settings()
    assert settings.model_name == "org/demo"
    assert settings.revision == "abc123"
    assert settings.model_path is None

    # Once the pinned revision is downloaded the record resolves to that snapshot, and the
    # engine key does not change (so the engine is not loaded twice).
    snap = _fake_snapshot(runtime.cache_manager, "org/demo", "abc123", "abc123", 10)
    runtime.model_registry.invalidate()
    key_after, _m, record_after = runtime.get_engine_key_for_tag("pinned:v1")
    assert record_before.snapshot_path != str(snap)
    assert record_after.snapshot_path == str(snap)
    assert key_after == key_before


# -------------------------------------------------------- M4 / H4: cache + download


def test_cached_models_do_not_double_count_symlinked_blobs(tmp_path: Path) -> None:
    hub = tmp_path / "hub"
    repo = hub / "models--org--demo"
    (repo / "blobs").mkdir(parents=True)
    blob = repo / "blobs" / "sha256-aaa"
    blob.write_bytes(b"x" * 4096)
    for commit in ("rev1", "rev2"):
        snap = repo / "snapshots" / commit
        snap.mkdir(parents=True)
        os.symlink(os.path.relpath(blob, snap), snap / "model.safetensors")
    (repo / "snapshots" / "rev1" / "config.json").write_bytes(b"{}")

    assert _dir_size_no_follow(repo) == 4096 + 2
    models = get_cached_models(cache_dir=str(hub))
    assert models == [
        {"repo_id": "org/demo", "path": str(repo), "size": format_bytes(4096 + 2)}
    ]


def test_cached_models_count_hardlinks_once(tmp_path: Path) -> None:
    root = tmp_path / "m"
    root.mkdir()
    (root / "a").write_bytes(b"y" * 100)
    os.link(root / "a", root / "b")
    assert _dir_size_no_follow(root) == 100


def test_download_snapshot_uses_allow_patterns_without_bin(tmp_path: Path) -> None:
    cache = CacheManager(forgeai_home=tmp_path / "home")
    with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "snap")) as dl:
        cache.download_snapshot("org/demo", revision="main")
    kwargs = dl.call_args.kwargs
    assert "ignore_patterns" not in kwargs
    patterns = kwargs["allow_patterns"]
    assert patterns == DOWNLOAD_ALLOW_PATTERNS
    for needed in ("*.safetensors", "*.json", "tokenizer*", "*.model", "*.tiktoken", "*.txt", "chat_template*"):
        assert needed in patterns
    assert not any(p in patterns for p in ("*.bin", "*.pt", "*.pth", "*"))
    assert loader_mod.DOWNLOAD_ALLOW_PATTERNS is DOWNLOAD_ALLOW_PATTERNS


# ------------------------------------------------------------------ M1: middleware


def _counter(status: str, method: str = "POST") -> float:
    return REQUESTS_TOTAL.labels(method=method, status=status)._value.get()


def test_429_has_request_id_cors_and_is_counted() -> None:
    class Limiter:
        def check(self, key: str):
            return False, 3

    settings = SimpleNamespace(request_id_header="X-Request-ID", cors_allow_origins=["http://ui.test"])
    app = create_app(settings=settings, rate_limiter=Limiter())
    before = _counter("429")
    active_before = ACTIVE_REQUESTS._value.get()

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": "req-429", "Origin": "http://ui.test"},
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

    resp = asyncio.run(run())
    assert resp.status_code == 429
    assert resp.headers["x-request-id"] == "req-429"
    assert resp.headers["retry-after"] == "3"
    assert resp.headers["access-control-allow-origin"] == "http://ui.test"
    assert resp.json()["error"]["request_id"] == "req-429"
    assert _counter("429") == before + 1
    assert ACTIVE_REQUESTS._value.get() == active_before


def test_429_without_client_request_id_uses_one_consistent_id() -> None:
    class Limiter:
        def check(self, key: str):
            return False, 1

    app = create_app(rate_limiter=Limiter())

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.get("/v1/models")

    resp = asyncio.run(run())
    assert resp.status_code == 429
    assert resp.headers["x-request-id"] == resp.json()["error"]["request_id"]


def test_401_is_counted_and_has_request_id() -> None:
    class Auth:
        def validate_api_key(self, key: str):
            return None

    app = create_app(enable_auth=True, auth_manager=Auth())
    before = _counter("401", "GET")

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            return await client.get("/v1/models", headers={"X-API-Key": "nope"})

    resp = asyncio.run(run())
    assert resp.status_code == 401
    assert "x-request-id" in resp.headers
    assert _counter("401", "GET") == before + 1


def _scope() -> dict[str, Any]:
    return {
        "type": "http",
        "method": "GET",
        "path": "/stream",
        "headers": [],
        "client": ("1.2.3.4", 5),
        "app": SimpleNamespace(state=SimpleNamespace(settings=None)),
    }


def test_streaming_latency_recorded_after_stream_completes() -> None:
    recorded: list[dict[str, Any]] = []
    sent: list[str] = []

    async def streaming_app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        sent.append("first-chunk")
        assert recorded == []  # not finalized at headers or first chunk
        await asyncio.sleep(0.12)
        scope["state"]["completion_tokens"] = 7
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    active_before = ACTIVE_REQUESTS._value.get()
    observed_active: list[float] = []

    async def run() -> None:
        mw = RequestLoggingMiddleware(streaming_app)

        async def receive() -> dict[str, Any]:
            await asyncio.sleep(10)
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                observed_active.append(ACTIVE_REQUESTS._value.get())
                names = [k.decode() for k, _ in message["headers"]]
                assert "x-request-id" in names

        await mw(_scope(), receive, send)  # type: ignore[arg-type]

    with patch.object(server_mod, "record_request", lambda **kw: recorded.append(kw)):
        asyncio.run(run())

    assert observed_active == [active_before + 1]
    assert len(recorded) == 1
    assert recorded[0]["duration"] >= 0.1
    assert recorded[0]["status"] == "200"
    assert recorded[0]["tokens"] == 7
    assert ACTIVE_REQUESTS._value.get() == active_before


def test_disconnect_finalizes_once_and_decrements_active() -> None:
    recorded: list[dict[str, Any]] = []

    async def hanging_app(scope: Any, receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"a", "more_body": True})
        while (await receive())["type"] != "http.disconnect":
            pass

    active_before = ACTIVE_REQUESTS._value.get()

    async def run() -> None:
        mw = RequestLoggingMiddleware(hanging_app)
        messages = iter([{"type": "http.disconnect"}])

        async def receive() -> dict[str, Any]:
            return next(messages)

        async def send(message: dict[str, Any]) -> None:
            return None

        await mw(_scope(), receive, send)  # type: ignore[arg-type]

    with patch.object(server_mod, "record_request", lambda **kw: recorded.append(kw)):
        asyncio.run(run())

    assert len(recorded) == 1
    assert recorded[0]["status"] == "200"
    assert ACTIVE_REQUESTS._value.get() == active_before


def test_exception_in_app_is_counted_as_500_and_reraised() -> None:
    recorded: list[dict[str, Any]] = []

    async def boom(scope: Any, receive: Any, send: Any) -> None:
        raise RuntimeError("boom")

    active_before = ACTIVE_REQUESTS._value.get()

    async def run() -> None:
        mw = RequestLoggingMiddleware(boom)

        async def send(message: dict[str, Any]) -> None:
            return None

        async def receive() -> dict[str, Any]:
            return {"type": "http.disconnect"}

        await mw(_scope(), receive, send)  # type: ignore[arg-type]

    with (
        patch.object(server_mod, "record_request", lambda **kw: recorded.append(kw)),
        pytest.raises(RuntimeError),
    ):
        asyncio.run(run())

    assert [r["status"] for r in recorded] == ["500"]
    assert ACTIVE_REQUESTS._value.get() == active_before


def test_middlewares_are_pure_asgi() -> None:
    from starlette.middleware.base import BaseHTTPMiddleware

    from forgeai.security.middleware import AuthMiddleware, RateLimitMiddleware

    for cls in (RequestLoggingMiddleware, AuthMiddleware, RateLimitMiddleware):
        assert not issubclass(cls, BaseHTTPMiddleware)
