from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.api.routes.chat import ChatCompletionRequest, create_chat_completion
from forgeai.api.routes.ollama import generate
from forgeai.api.runtime import SharedRuntimeAdapter
from forgeai.api.schemas.ollama import OllamaGenerateRequest
from forgeai.api.server import create_app
from forgeai.core.backends.vllm_backend import VLLMBackend
from forgeai.core.config import DevToolSettings
from forgeai.core.engine import (
    EngineKey,
    EngineLease,
    EngineQueueFullError,
    EngineState,
    EngineStatus,
)
from forgeai.models.manifest import ForgeAIManifest
from forgeai.models.registry import ModelRegistry


class FakeEngine:
    def __init__(
        self,
        model_name: str = "gemma-2-9b-it:latest",
        supports_embed: bool = True,
        raise_on_build_prompt: bool = False,
    ) -> None:
        self.is_running = True
        self.supports_streaming = True
        self.supports_embed = supports_embed
        self.raise_on_build_prompt = raise_on_build_prompt
        self.settings = SimpleNamespace(model_name=model_name)
        self.last_prompt = None
        self.last_messages = None
        self.last_top_k = None

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        if self.raise_on_build_prompt:
            raise RuntimeError("Simulated prompt building failure.")
        self.last_messages = messages
        return f"<prompt>{messages}</prompt>"

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ) -> SimpleNamespace:
        self.last_prompt = prompt
        self.last_top_k = top_k
        return SimpleNamespace(
            text="generated text response",
            prompt_tokens=8,
            completion_tokens=4,
            total_tokens=12,
            finish_reason="stop",
        )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ):
        self.last_prompt = prompt
        self.last_top_k = top_k
        yield "hello "
        yield "world"

    async def embed(self, input_texts: list[str]) -> list[list[float]]:
        if not self.supports_embed:
            raise NotImplementedError("Embedding not supported by this mock engine.")
        return [[0.1, 0.2, 0.3] for _ in input_texts]


class FakeEngineManager:
    def __init__(self, engine: Any = None) -> None:
        self.engine = engine or FakeEngine()
        self.acquire_calls = []
        self.release_calls = []
        self.stopped_keys = []
        self.failed_keys = []
        self._shutting_down = False

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    async def acquire(self, key: EngineKey, keep_alive: Any = None) -> EngineLease:
        if self._shutting_down:
            raise RuntimeError("EngineManager is shutting down.")
        self.acquire_calls.append((key, keep_alive))
        return EngineLease(
            key=key,
            engine=self.engine,
            state=EngineState.READY,
            ref_count=1,
        )

    async def release(self, key: EngineKey, keep_alive: Any = None) -> None:
        self.release_calls.append((key, keep_alive))

    async def list_async(self) -> list[EngineStatus]:
        return [
            EngineStatus(
                key=EngineKey(repo_id="gemma-2-9b-it:latest"),
                state=EngineState.READY,
                ref_count=1,
                created_at=1000.0,
                last_accessed_at=1000.0,
                remaining_keep_alive_seconds=None,
            )
        ]

    async def stop(self, key_or_tag: Any, drain_timeout: float = 10.0) -> None:
        self.stopped_keys.append(key_or_tag)

    async def mark_engine_failed(self, key: EngineKey, reason: str = "") -> None:
        self.failed_keys.append((key, reason))

    async def shutdown(self) -> None:
        self._shutting_down = True


class FakeCacheManager:
    def __init__(self, hub_dir: Path) -> None:
        self.hub_dir = hub_dir
        self.download_snapshot_calls = []

    def get_snapshot_path(self, repo_id: str, revision: str = "main") -> Path | None:
        snap = self.hub_dir / repo_id.replace("/", "--") / "snapshots" / "main"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text("{}", encoding="utf-8")
        (snap / "model.safetensors").write_text("weights", encoding="utf-8")
        return snap

    def download_snapshot(self, repo_id: str, revision: str = "main", token: str | None = None) -> str:
        self.download_snapshot_calls.append({"repo_id": repo_id, "revision": revision, "token": token})
        if "fail_download" in repo_id:
            raise RuntimeError("Simulated network/download failure.")
        snap = self.get_snapshot_path(repo_id, revision)
        return str(snap)


class OllamaApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        manifests_dir = self.base_dir / "manifests"
        manifests_dir.mkdir()

        self.model_registry = ModelRegistry(manifests_dir=manifests_dir)
        self.cache_manager = FakeCacheManager(hub_dir=self.base_dir / "hub")
        self.fake_engine = FakeEngine()
        self.engine_manager = FakeEngineManager(engine=self.fake_engine)

        self.manifest = ForgeAIManifest(
            name="gemma-2-9b-it:latest",
            model="google/gemma-2-9b-it",
            source_kind="huggingface",
        )
        self.model_registry.register_manifest(self.manifest, cache_manager=self.cache_manager)

        self.runtime_adapter = SharedRuntimeAdapter(
            engine_manager=self.engine_manager,
            model_registry=self.model_registry,
            cache_manager=self.cache_manager,
        )

        self.app = create_app(runtime_adapter=self.runtime_adapter)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def async_client(self, app=None):
        target_app = app or self.app
        transport = httpx.ASGITransport(app=target_app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def test_streaming_lease_lifetime_held_during_iteration(self) -> None:
        """Verify stream generation completes and releases lease in finally."""
        async with self.async_client() as client:
            res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Test", "stream": True},
            )

        self.assertEqual(res.status_code, 200)
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
        self.assertTrue(len(lines) >= 2)
        self.assertEqual(len(self.engine_manager.acquire_calls), 1)
        self.assertEqual(len(self.engine_manager.release_calls), 1)

    async def test_top_k_option_forwarding(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/generate",
                json={
                    "model": "gemma-2-9b-it:latest",
                    "prompt": "Hello",
                    "options": {"top_k": 42},
                    "stream": False,
                },
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.fake_engine.last_top_k, 42)

    async def test_unsupported_suffix_and_template_rejected(self) -> None:
        async with self.async_client() as client:
            suffix_res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Hi", "suffix": "end"},
            )
            template_res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Hi", "template": "custom"},
            )

        self.assertEqual(suffix_res.status_code, 400)
        self.assertIn("Suffix parameter is unsupported", suffix_res.json()["error"])

        self.assertEqual(template_res.status_code, 400)
        self.assertIn("custom Jinja template override is unsupported", template_res.json()["error"])

    async def test_unsupported_images_rejected(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Look", "images": ["base64img"]},
            )

        self.assertEqual(res.status_code, 400)
        self.assertIn("Multimodal image inputs are deferred", res.json()["error"])

    async def test_empty_model_validation_error(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/generate",
                json={"model": "   ", "prompt": "Hi"},
            )

        self.assertEqual(res.status_code, 422)

    async def test_embed_dimensions_rejected(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/embed",
                json={"model": "gemma-2-9b-it:latest", "input": "text", "dimensions": 128},
            )

        self.assertEqual(res.status_code, 400)
        self.assertIn("dimensions", res.json()["error"])

    async def test_embed_truncate_rejected(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/embed",
                json={"model": "gemma-2-9b-it:latest", "input": "text", "truncate": True},
            )

        self.assertEqual(res.status_code, 400)
        self.assertIn("truncate", res.json()["error"])

    async def test_embed_options_rejected(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/embed",
                json={"model": "gemma-2-9b-it:latest", "input": "text", "options": {"some": "opt"}},
            )

        self.assertEqual(res.status_code, 400)
        self.assertIn("options", res.json()["error"])

    async def test_pull_uses_download_snapshot_and_tagged_repo_parsing(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/pull",
                json={"model": "org/model:custom-v1", "stream": True},
            )

        self.assertEqual(res.status_code, 200)
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
        self.assertEqual(lines[-1]["status"], "success")
        self.assertIn("digest", lines[-1])

        self.assertEqual(len(self.cache_manager.download_snapshot_calls), 1)
        self.assertEqual(
            self.cache_manager.download_snapshot_calls[0],
            {"repo_id": "org/model", "revision": "main", "token": None},
        )

        reg_manifest = self.model_registry.get_manifest("org/model:custom-v1")
        self.assertEqual(reg_manifest.model, "org/model")

    async def test_pull_emits_no_verifying_sha256_status(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/pull",
                json={"model": "org/model:v2", "stream": True},
            )

        self.assertEqual(res.status_code, 200)
        statuses = [json.loads(line).get("status") for line in res.text.strip().split("\n") if line.strip()]
        self.assertNotIn("verifying sha256 digest", statuses)
        self.assertIn("downloading weights", statuses)
        self.assertIn("writing manifest", statuses)
        self.assertIn("success", statuses)

    async def test_pull_failure_preserves_atomic_manifest(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/pull",
                json={"model": "org/fail_download:latest", "stream": True},
            )

        self.assertEqual(res.status_code, 200)
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line.strip()]
        self.assertIn("error", lines[-1])
        self.assertIn("Pull failed", lines[-1]["error"])

        with self.assertRaises(KeyError):
            self.model_registry.get_manifest("org/fail_download:latest")

    async def test_delete_uses_exact_engine_key(self) -> None:
        async with self.async_client() as client:
            res = await client.request(
                "DELETE",
                "/api/delete",
                json={"model": "gemma-2-9b-it:latest"},
            )

        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(self.engine_manager.stopped_keys), 1)
        key = self.engine_manager.stopped_keys[0]
        self.assertIsInstance(key, EngineKey)

    async def test_ps_uses_public_tag(self) -> None:
        async with self.async_client() as client:
            res = await client.get("/api/ps")

        self.assertEqual(res.status_code, 200)
        m = res.json()["models"][0]
        self.assertEqual(m["name"], "gemma-2-9b-it:latest")
        self.assertEqual(m["size_vram"], 0)
        self.assertEqual(m["expires_at"], "2100-01-01T00:00:00Z")

    async def test_tags_modified_at_mtime(self) -> None:
        async with self.async_client() as client:
            res = await client.get("/api/tags")

        self.assertEqual(res.status_code, 200)
        m = res.json()["models"][0]
        self.assertIn("modified_at", m)
        self.assertIn("T", m["modified_at"])

    async def test_show_modelfile_and_manifest(self) -> None:
        async with self.async_client() as client:
            res = await client.post("/api/show", json={"model": "gemma-2-9b-it:latest"})

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("ForgeAI Local Manifest Replacement", data["modelfile"])
        self.assertIn("forgeai_manifest", data)
        self.assertEqual(data["forgeai_manifest"]["name"], "gemma-2-9b-it:latest")

    async def test_typed_pre_stream_error_headers(self) -> None:
        class QueueFullManager(FakeEngineManager):
            async def acquire(self, key, keep_alive=None):
                raise EngineQueueFullError("Queue full")

        app = create_app(
            runtime_adapter=SharedRuntimeAdapter(
                engine_manager=QueueFullManager(),
                model_registry=self.model_registry,
                cache_manager=self.cache_manager,
            )
        )

        async with self.async_client(app) as client:
            res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Hi"},
            )

        self.assertEqual(res.status_code, 429)
        self.assertEqual(res.headers["X-ForgeAI-Error-Code"], "ERR_QUEUE_FULL")
        self.assertEqual(res.json(), {"error": "Queue full"})

    async def test_openai_sse_stream_lease_held_through_done(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "gemma-2-9b-it:latest",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

        self.assertEqual(res.status_code, 200)
        lines = [line.strip() for line in res.text.strip().split("\n") if line.strip()]
        self.assertTrue(lines[-1].endswith("[DONE]"))
        self.assertEqual(len(self.engine_manager.acquire_calls), 1)
        self.assertEqual(len(self.engine_manager.release_calls), 1)

    async def test_setup_exception_releases_lease(self) -> None:
        """Verify lease is released when build_prompt raises an exception after acquire."""
        failing_engine = FakeEngine(raise_on_build_prompt=True)
        mgr = FakeEngineManager(engine=failing_engine)
        adapter = SharedRuntimeAdapter(
            engine_manager=mgr,
            model_registry=self.model_registry,
            cache_manager=self.cache_manager,
        )

        req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_adapter=adapter)))
        body = OllamaGenerateRequest(model="gemma-2-9b-it:latest", prompt="test", stream=False)

        with self.assertRaises(RuntimeError):
            await generate(req, body)

        self.assertEqual(len(mgr.acquire_calls), 1)
        self.assertEqual(len(mgr.release_calls), 1)

    async def test_openai_stream_setup_error_releases_lease_before_response(self) -> None:
        """Verify OpenAI SSE pre-stream setup error releases lease before returning response."""
        failing_engine = FakeEngine(raise_on_build_prompt=True)
        mgr = FakeEngineManager(engine=failing_engine)
        adapter = SharedRuntimeAdapter(
            engine_manager=mgr,
            model_registry=self.model_registry,
            cache_manager=self.cache_manager,
        )

        req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(runtime_adapter=adapter)))
        body = ChatCompletionRequest(
            model="gemma-2-9b-it:latest",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        with self.assertRaises(RuntimeError):
            await create_chat_completion(req, body)

        self.assertEqual(len(mgr.acquire_calls), 1)
        self.assertEqual(len(mgr.release_calls), 1)

    async def test_unsupported_controls_do_not_acquire_engine(self) -> None:
        """Verify unsupported controls return 400 without acquiring an engine lease."""
        async with self.async_client() as client:
            img_res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Hi", "images": ["img"]},
            )
            suf_res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Hi", "suffix": "suf"},
            )
            tmpl_res = await client.post(
                "/api/generate",
                json={"model": "gemma-2-9b-it:latest", "prompt": "Hi", "template": "tmpl"},
            )

        self.assertEqual(img_res.status_code, 400)
        self.assertEqual(suf_res.status_code, 400)
        self.assertEqual(tmpl_res.status_code, 400)
        self.assertEqual(len(self.engine_manager.acquire_calls), 0)

    async def test_revision_forwarding_to_settings_and_vllm_kwargs(self) -> None:
        """Verify EngineKey.revision reaches DevToolSettings and vLLM constructor kwargs."""
        key = EngineKey(repo_id="google/gemma-2-9b-it", revision="v1.2.3")
        settings = key.to_settings()
        self.assertEqual(settings.revision, "v1.2.3")

        vllm_kwargs = settings.to_vllm_kwargs()
        self.assertEqual(vllm_kwargs.get("revision"), "v1.2.3")

        backend = VLLMBackend(settings)
        async_kwargs = backend._build_async_engine_args_kwargs()
        self.assertEqual(async_kwargs.get("revision"), "v1.2.3")

    async def test_resource_safe_stop_control(self) -> None:
        """Verify POST /api/generate with keep_alive=0 and empty prompt stops engine without acquire/generate calls."""
        async with self.async_client() as client:
            res = await client.post(
                "/api/generate",
                json={
                    "model": "gemma-2-9b-it:latest",
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
            )

        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["done"])
        self.assertEqual(data["done_reason"], "stop")
        self.assertEqual(data["response"], "")
        self.assertEqual(len(self.engine_manager.acquire_calls), 0)
        self.assertEqual(len(self.engine_manager.stopped_keys), 1)
        self.assertIsInstance(self.engine_manager.stopped_keys[0], EngineKey)

        # Missing model yields 404
        async with self.async_client() as client:
            res_missing = await client.post(
                "/api/generate",
                json={
                    "model": "nonexistent:latest",
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
            )
        self.assertEqual(res_missing.status_code, 404)

        # Verify stop does not unregister the model
        manifest = self.model_registry.get_manifest("gemma-2-9b-it:latest")
        self.assertEqual(manifest.name, "gemma-2-9b-it:latest")


if __name__ == "__main__":
    unittest.main()
