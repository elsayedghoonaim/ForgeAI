"""Phase 4: API and CLI correctness (scanner, timeouts, teardown, error codes, unload, batch)."""

from __future__ import annotations

import asyncio
import collections
import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_ollama_api as base  # noqa: E402
from forgeai.api.routes.chat import ChatMessage  # noqa: E402
from forgeai.benchmarking.runner import _spawn_server_process  # noqa: E402
from forgeai.cli.commands.batch import load_prompts, process_prompts  # noqa: E402
from forgeai.cli.main import app  # noqa: E402
from forgeai.cli.runtime import DaemonClient, DaemonClientError, resolve_base_url  # noqa: E402
from forgeai.models.loader import CacheManager, SecurityBlockError  # noqa: E402
from forgeai.models.safety_scanner import _scan_pickle_bytes, scan_model_weights  # noqa: E402

runner = CliRunner()


def _write_bin(path: Path, payload: bytes) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", payload)


class ScannerAllowlistTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_allowlisted_globals_are_safe(self) -> None:
        payload = pickle.dumps(collections.OrderedDict(a=1), protocol=2)
        self.assertEqual(_scan_pickle_bytes(__import__("io").BytesIO(payload), "m.bin"), [])
        torch_payload = b"ctorch._utils\n_rebuild_tensor_v2\n."
        self.assertEqual(_scan_pickle_bytes(__import__("io").BytesIO(torch_payload), "m.bin"), [])

    def test_unknown_global_is_unsafe(self) -> None:
        for payload in (b"cbuiltins\neval\n.", b"cevil_pkg\nthing\n.", b"ctorch\nload\n."):
            warnings = _scan_pickle_bytes(__import__("io").BytesIO(payload), "m.bin")
            self.assertTrue(any("Unsafe pickle" in w for w in warnings), payload)

    def test_unparseable_pickle_is_unsafe(self) -> None:
        warnings = _scan_pickle_bytes(__import__("io").BytesIO(b"\x80\x04\xff\xfenot a pickle"), "m.bin")
        self.assertTrue(any("Unsafe pickle" in w for w in warnings))
        _write_bin(self.dir / "model.bin", b"\x80\x04\xff\xfe" * 40)
        self.assertFalse(scan_model_weights(str(self.dir / "model.bin"))["safe"])

    def test_unresolvable_stack_global_is_unsafe(self) -> None:
        # protocol-4 STACK_GLOBAL whose operands are not plain string pushes
        payload = b"\x80\x04N\x94N\x94\x93."
        warnings = _scan_pickle_bytes(__import__("io").BytesIO(payload), "m.bin")
        self.assertTrue(any("Unsafe pickle" in w for w in warnings))

    def test_many_safe_bin_shards_do_not_fail(self) -> None:
        payload = pickle.dumps(collections.OrderedDict(w=[1.0]), protocol=2)
        for i in range(6):
            _write_bin(self.dir / f"pytorch_model-{i}.bin", payload)
        result = scan_model_weights(str(self.dir))
        self.assertTrue(result["safe"], result["warnings"])
        self.assertEqual(result["score"], 100)

    def test_one_bad_shard_among_safe_ones_blocks(self) -> None:
        good = pickle.dumps({"w": [1.0]})
        for i in range(3):
            _write_bin(self.dir / f"good-{i}.bin", good)
        _write_bin(self.dir / "bad.bin", b"cposix\nsystem\n(S'echo'\ntR.")
        self.assertFalse(scan_model_weights(str(self.dir))["safe"])


def _make_hf_repo(root: Path) -> tuple[CacheManager, Path, Path]:
    """Build a cache with two snapshots sharing one blob. Returns (manager, snap_a, snap_b)."""
    manager = CacheManager(forgeai_home=root / "fh", hf_home=root / "hf")
    repo = manager.get_repo_dir("org/model")
    (repo / "blobs").mkdir(parents=True)
    (repo / "refs").mkdir()
    shared = repo / "blobs" / "shared"
    shared.write_text("{}")
    only_a = repo / "blobs" / "onlya"
    only_a.write_text("weights")
    snaps = []
    for commit in ("aaa", "bbb"):
        snap = repo / "snapshots" / commit
        snap.mkdir(parents=True)
        os.symlink(shared, snap / "config.json")
        snaps.append(snap)
    os.symlink(only_a, snaps[0] / "model.safetensors")
    (repo / "refs" / "main").write_text("aaa")
    (repo / "refs" / "other").write_text("bbb")
    return manager, snaps[0], snaps[1]


class PurgeSnapshotTests(unittest.TestCase):
    def test_purge_removes_snapshot_blobs_and_refs_but_keeps_shared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, snap_a, snap_b = _make_hf_repo(Path(tmp))
            repo = manager.get_repo_dir("org/model")
            manager.purge_snapshot(snap_a)
            self.assertFalse(snap_a.exists())
            self.assertFalse((repo / "blobs" / "onlya").exists())
            self.assertFalse((repo / "refs" / "main").exists())
            self.assertTrue((repo / "blobs" / "shared").exists())
            self.assertTrue(snap_b.exists())
            self.assertTrue((repo / "refs" / "other").exists())

    def test_scan_failure_purges_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, snap_a, _ = _make_hf_repo(Path(tmp))
            repo = manager.get_repo_dir("org/model")
            bad = repo / "blobs" / "badpickle"
            with zipfile.ZipFile(bad, "w") as z:
                z.writestr("a/data.pkl", b"cposix\nsystem\n(S'echo'\ntR.")
            os.symlink(bad, snap_a / "pytorch_model.bin")
            with self.assertRaises(SecurityBlockError):
                manager.scan_snapshot_or_purge(snap_a, "org/model")
            self.assertFalse(snap_a.exists())
            self.assertFalse(bad.exists())
            self.assertFalse((repo / "refs" / "main").exists())


class DownloadPatternTests(unittest.TestCase):
    def _download(self, **kwargs: object) -> list[str]:
        mock_hf = MagicMock()
        mock_hf.snapshot_download.return_value = "/tmp/x"
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
            manager = CacheManager(forgeai_home=Path(tmp) / "fh", hf_home=Path(tmp) / "hf")
            manager.download_snapshot("org/model", **kwargs)  # type: ignore[arg-type]
        patterns: list[str] = mock_hf.snapshot_download.call_args.kwargs["allow_patterns"]
        return patterns

    def test_py_files_excluded_by_default(self) -> None:
        patterns = self._download()
        self.assertNotIn("*.py", patterns)
        self.assertNotIn("*.bin", patterns)

    def test_py_files_included_only_with_trust_remote_code(self) -> None:
        self.assertIn("*.py", self._download(trust_remote_code=True))


class PullApiTests(unittest.IsolatedAsyncioTestCase):
    setUp = base.OllamaApiTests.setUp
    tearDown = base.OllamaApiTests.tearDown
    async_client = base.OllamaApiTests.async_client

    async def test_blocked_scan_returns_error_and_does_not_register(self) -> None:
        self.cache_manager.scan_error = SecurityBlockError("SECURITY BLOCK: nope")
        async with self.async_client() as client:
            res = await client.post("/api/pull", json={"model": "org/evil:latest", "stream": False})
            res_stream = await client.post("/api/pull", json={"model": "org/evil:latest", "stream": True})
        self.assertEqual(res.status_code, 403)
        self.assertIn("SECURITY BLOCK", res.json()["error"])
        last = json.loads(res_stream.text.strip().splitlines()[-1])
        self.assertIn("SECURITY BLOCK", last["error"])
        with self.assertRaises(KeyError):
            self.model_registry.get_manifest("org/evil:latest")

    async def test_pull_scans_and_trust_remote_code_flows_to_download(self) -> None:
        async with self.async_client() as client:
            res = await client.post(
                "/api/pull",
                json={"model": "org/rc:latest", "stream": False, "trust_remote_code": True},
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(self.cache_manager.scan_calls, ["org/rc"])
        self.assertTrue(self.cache_manager.download_snapshot_calls[0]["trust_remote_code"])
        self.assertTrue(self.model_registry.get_manifest("org/rc:latest").engine_settings.trust_remote_code)


class ApiCorrectnessTests(unittest.IsolatedAsyncioTestCase):
    setUp = base.OllamaApiTests.setUp
    tearDown = base.OllamaApiTests.tearDown
    async_client = base.OllamaApiTests.async_client

    async def _post(self, path: str, payload: dict) -> httpx.Response:
        async with self.async_client() as client:
            return await client.post(path, json=payload)

    async def test_bad_tag_is_400_not_500(self) -> None:
        for path, payload in (
            ("/api/generate", {"model": "../etc/passwd", "prompt": "hi", "stream": False}),
            ("/api/chat", {"model": "a\\b", "messages": [{"role": "user", "content": "x"}], "stream": False}),
            ("/api/embed", {"model": "x/../y", "input": "hi"}),
            ("/v1/chat/completions", {"model": "../x", "messages": [{"role": "user", "content": "x"}]}),
        ):
            res = await self._post(path, payload)
            self.assertEqual(res.status_code, 400, (path, res.text))

    async def test_unknown_model_is_404(self) -> None:
        res = await self._post("/api/generate", {"model": "nope:latest", "prompt": "hi", "stream": False})
        self.assertEqual(res.status_code, 404)

    async def test_embed_failure_is_500(self) -> None:
        async def boom(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("engine exploded")

        self.fake_engine.embed = boom  # type: ignore[method-assign]
        res = await self._post("/api/embed", {"model": "gemma-2-9b-it:latest", "input": "hi"})
        self.assertEqual(res.status_code, 500)

    async def test_embed_prompt_eval_count_uses_tokenizer(self) -> None:
        self.fake_engine.tokenizer = SimpleNamespace(encode=lambda t: list(range(7)))  # type: ignore[attr-defined]
        res = await self._post("/api/embed", {"model": "gemma-2-9b-it:latest", "input": ["a b", "c"]})
        self.assertEqual(res.json()["prompt_eval_count"], 14)

    async def test_embed_prompt_eval_count_falls_back_to_word_count(self) -> None:
        res = await self._post("/api/embed", {"model": "gemma-2-9b-it:latest", "input": ["a b c", "d"]})
        self.assertEqual(res.json()["prompt_eval_count"], 4)

    async def test_unload_shape_never_loads_regardless_of_stream(self) -> None:
        for stream in (True, False):
            res = await self._post(
                "/api/generate",
                {"model": "gemma-2-9b-it:latest", "prompt": "", "keep_alive": 0, "stream": stream},
            )
            self.assertEqual(res.status_code, 200, res.text)
            self.assertTrue(res.json()["done"])
        # default stream (True) with no explicit field
        res = await self._post("/api/generate", {"model": "gemma-2-9b-it:latest", "keep_alive": "0"})
        self.assertEqual(res.status_code, 200)
        res = await self._post(
            "/api/chat", {"model": "gemma-2-9b-it:latest", "messages": [], "keep_alive": 0}
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["done"])
        self.assertEqual(len(self.engine_manager.acquire_calls), 0)
        self.assertEqual(len(self.engine_manager.stopped_keys), 4)

    async def test_unload_endpoint_reports_loaded_state(self) -> None:
        key, _, _ = self.runtime_adapter.get_engine_key_for_tag("gemma-2-9b-it:latest")

        async def loaded() -> list:
            return [SimpleNamespace(key=key)]

        res = await self._post("/api/unload", {"model": "gemma-2-9b-it:latest"})
        self.assertEqual(res.json(), {"model": "gemma-2-9b-it:latest", "unloaded": False})

        self.engine_manager.list_async = loaded  # type: ignore[method-assign]
        res = await self._post("/api/unload", {"model": "gemma-2-9b-it:latest"})
        self.assertEqual(res.json()["unloaded"], True)
        self.assertEqual(len(self.engine_manager.acquire_calls), 0)

        self.assertEqual((await self._post("/api/unload", {"model": "nope:latest"})).status_code, 404)
        self.assertEqual((await self._post("/api/unload", {})).status_code, 400)

    async def test_openai_content_parts_and_null(self) -> None:
        res = await self._post(
            "/v1/chat/completions",
            {
                "model": "gemma-2-9b-it:latest",
                "messages": [
                    {"role": "system", "content": None},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello "},
                            {"type": "image_url", "image_url": {"url": "http://x"}},
                            {"type": "text", "text": "there"},
                        ],
                    },
                ],
            },
        )
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(self.fake_engine.last_messages[1]["content"], "Hello there")
        self.assertEqual(self.fake_engine.last_messages[0]["content"], "")
        self.assertEqual(ChatMessage(role="user", content=None).content, "")  # type: ignore[arg-type]

    async def test_openai_stream_first_chunk_role_and_final_finish_reason(self) -> None:
        res = await self._post(
            "/v1/chat/completions",
            {"model": "gemma-2-9b-it:latest", "messages": [{"role": "user", "content": "x"}], "stream": True},
        )
        frames = [json.loads(line[6:]) for line in res.text.splitlines() if line.startswith("data: {")]
        self.assertEqual(frames[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertIsNone(frames[0]["choices"][0]["finish_reason"])
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "stop")

    async def test_models_path_route_accepts_slashes(self) -> None:
        reg = self.model_registry
        reg.register_manifest(
            base.ForgeAIManifest(name="org/model:v1", model="org/model", source_kind="huggingface"),
            cache_manager=self.cache_manager,
        )
        async with self.async_client() as client:
            res = await client.get("/v1/models/org/model:v1")
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(res.json()["id"], "org/model:v1")


class TimeoutTests(unittest.TestCase):
    def test_timeout_config_and_env_overrides(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            for var in ("FORGEAI_CONNECT_TIMEOUT", "FORGEAI_REQUEST_TIMEOUT", "FORGEAI_LONG_TIMEOUT"):
                os.environ.pop(var, None)
            client = DaemonClient(base_url="http://x:1")
            quick, long = client.build_timeout(False), client.build_timeout(True)
            self.assertEqual((quick.connect, quick.read), (5.0, 30.0))
            self.assertEqual((long.connect, long.read), (5.0, None))
        with patch.dict(
            os.environ,
            {"FORGEAI_CONNECT_TIMEOUT": "2", "FORGEAI_REQUEST_TIMEOUT": "7", "FORGEAI_LONG_TIMEOUT": "600"},
        ):
            client = DaemonClient(base_url="http://x:1")
            self.assertEqual(client.build_timeout(True).read, 600.0)
            self.assertEqual(client.build_timeout(False).read, 7.0)
            self.assertEqual(client.build_timeout(False).connect, 2.0)

    def test_read_timeout_reports_slow_daemon_not_down(self) -> None:
        client = DaemonClient(base_url="http://x:1")
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.request.side_effect = httpx.ReadTimeout("slow")
            with self.assertRaises(DaemonClientError) as ctx:
                client.request("GET", "/api/ps")
        self.assertIn("did not respond within 30 s", str(ctx.exception))
        self.assertNotIn("Is the server running", str(ctx.exception))

    def test_connect_timeout_and_refused_report_connection(self) -> None:
        client = DaemonClient(base_url="http://x:1")
        for exc in (httpx.ConnectTimeout("t"), httpx.ConnectError("refused")):
            with patch("httpx.Client") as mock_client:
                mock_client.return_value.__enter__.return_value.request.side_effect = exc
                with self.assertRaises(DaemonClientError) as ctx:
                    client.request("GET", "/api/ps")
            self.assertIn("Could not connect", str(ctx.exception))

    def test_stream_has_no_read_timeout(self) -> None:
        client = DaemonClient(base_url="http://x:1")
        with patch("httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.stream.side_effect = httpx.ReadTimeout("t")
            with self.assertRaises(DaemonClientError):
                list(client.stream("POST", "/api/generate", json_data={}))
        self.assertIsNone(mock_client.call_args.kwargs["timeout"].read)


class BaseUrlTests(unittest.TestCase):
    def test_ipv6_hosts(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FORGEAI_PORT", None)
            os.environ.pop("forgeai_port", None)
            self.assertEqual(resolve_base_url(host="::1"), "http://[::1]:11434")
            self.assertEqual(resolve_base_url(host="[::1]"), "http://[::1]:11434")
            self.assertEqual(resolve_base_url(host="[::1]:8080"), "http://[::1]:8080")
            self.assertEqual(resolve_base_url(host="::1", port=9000), "http://[::1]:9000")
            self.assertEqual(resolve_base_url(base_url="::1"), "http://[::1]")
            self.assertEqual(resolve_base_url(host="localhost:8080"), "http://localhost:8080")


class ProcessGroupTeardownTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_teardown_kills_grandchildren(self) -> None:
        from forgeai.benchmarking.runner import SequentialBenchmarkRunner

        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "child.pid"
            script = f"sleep 300 & echo $! > {pidfile}; wait"
            proc = _spawn_server_process(["sh", "-c", script])
            for _ in range(50):
                if pidfile.exists() and pidfile.read_text().strip():
                    break
                time.sleep(0.1)
            child_pid = int(pidfile.read_text())
            self.assertEqual(os.getpgid(proc.pid), proc.pid)  # own group
            runner_obj = SequentialBenchmarkRunner.__new__(SequentialBenchmarkRunner)
            runner_obj._owns_process_group = True
            runner_obj.shutdown_timeout_seconds = 5.0
            runner_obj._safely_teardown_process(proc)
            for _ in range(50):
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            with self.assertRaises(ProcessLookupError):
                os.kill(child_pid, 0)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups")
    def test_sigkill_escalation_after_timeout(self) -> None:
        from forgeai.benchmarking.runner import SequentialBenchmarkRunner

        proc = _spawn_server_process(["sh", "-c", "trap '' TERM; sleep 300 & wait"])
        time.sleep(0.3)
        runner_obj = SequentialBenchmarkRunner.__new__(SequentialBenchmarkRunner)
        runner_obj._owns_process_group = True
        runner_obj.shutdown_timeout_seconds = 0.5
        runner_obj._safely_teardown_process(proc)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(proc.returncode, -signal.SIGKILL)

    def test_nvidia_smi_calls_have_timeout(self) -> None:
        from forgeai.benchmarking import runner as runner_mod

        with patch.object(subprocess, "run") as mock_run:
            mock_run.return_value = SimpleNamespace(stdout="1024\n")
            runner_mod.default_gpu_memory_used_query()
        self.assertIsNotNone(mock_run.call_args.kwargs.get("timeout"))


class GpuShutdownTests(unittest.TestCase):
    def test_nvml_shutdown_called_when_query_raises(self) -> None:
        from forgeai.utils.gpu import detect_gpus

        fake = MagicMock()
        fake.nvmlDeviceGetCount.side_effect = RuntimeError("boom")
        with patch.dict(sys.modules, {"pynvml": fake}):
            detect_gpus()
        fake.nvmlShutdown.assert_called_once()


class CliThinClientTests(unittest.TestCase):
    def test_ps_handles_null_size_vram(self) -> None:
        with patch.object(DaemonClient, "request") as req:
            req.return_value = {"models": [{"name": "m:latest", "size_vram": None, "details": {}}]}
            res = runner.invoke(app, ["ps"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Unknown", res.output)

    def test_load_prompts_validates_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "in.jsonl"
            path.write_text(
                '{"prompt": "ok"}\n[1,2]\n"str"\n{"prompt": 5}\nnot json\n\n{"other": "x"}\n{"prompt": "two"}\n',
                encoding="utf-8",
            )
            prompts, problems = load_prompts(path, "prompt")
        self.assertEqual(prompts, [(1, "ok"), (8, "two")])
        self.assertEqual(len(problems), 5)

    def test_batch_runs_concurrently_up_to_batch_size(self) -> None:
        state = {"in_flight": 0, "max": 0}

        async def handler(request: httpx.Request) -> httpx.Response:
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(0.05)
            state["in_flight"] -= 1
            body = json.loads(request.content)
            if body["prompt"] == "bad":
                return httpx.Response(500, json={"error": "kaboom"})
            return httpx.Response(
                200,
                json={"response": body["prompt"].upper(), "prompt_eval_count": 2, "eval_count": 3, "done_reason": "stop"},
            )

        async def go() -> list[dict]:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler), base_url="http://d"
            ) as http:
                prompts = [(i + 1, "bad" if i == 3 else f"p{i}") for i in range(8)]
                return await process_prompts(
                    http, "m:latest", prompts, max_tokens=8, temperature=0.0, batch_size=4
                )

        results = asyncio.run(go())
        self.assertEqual(state["max"], 4)
        self.assertEqual([r["line"] for r in results], list(range(1, 9)))
        self.assertEqual(results[0]["output"], "P0")
        self.assertEqual(results[0]["tokens"], 5)
        self.assertIn("kaboom", results[3]["error"])

    def test_chat_is_thin_client_with_history(self) -> None:
        calls: list[dict] = []

        def fake_stream(self: DaemonClient, method: str, path: str, json_data: object = None, params: object = None):
            calls.append({"path": path, "body": json_data})
            reply = f"answer{len(calls)}"
            yield {"message": {"role": "assistant", "content": reply}}

        with (
            patch.object(DaemonClient, "stream", fake_stream),
            patch("builtins.input", side_effect=["one", "two", "/exit"]),
            patch("forgeai.core.engine.DevToolEngine") as engine_cls,
        ):
            res = runner.invoke(app, ["chat", "m:latest", "--system", "be brief"])
        self.assertEqual(res.exit_code, 0, res.output)
        engine_cls.assert_not_called()
        self.assertEqual([c["path"] for c in calls], ["/api/chat", "/api/chat"])
        second = calls[1]["body"]["messages"]
        self.assertEqual(
            [m["role"] for m in second], ["system", "user", "assistant", "user"]
        )
        self.assertEqual(second[2]["content"], "answer1")

    def test_ctrl_c_during_stream_cancels_and_returns_to_prompt(self) -> None:
        closed = {"v": False}

        class Gen:
            def __init__(self) -> None:
                self.n = 0

            def __iter__(self) -> Gen:
                return self

            def __next__(self) -> dict:
                self.n += 1
                if self.n == 1:
                    return {"message": {"content": "part"}}
                raise KeyboardInterrupt

            def close(self) -> None:
                closed["v"] = True

        bodies: list[dict] = []

        def fake_stream(self: DaemonClient, method: str, path: str, json_data: dict | None = None, params: object = None):
            bodies.append(json_data or {})
            if len(bodies) == 1:
                return Gen()
            return iter([{"message": {"content": "fine"}}])

        with (
            patch.object(DaemonClient, "stream", fake_stream),
            patch("builtins.input", side_effect=["first", "second", "/bye"]),
        ):
            res = runner.invoke(app, ["run", "m:latest"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertTrue(closed["v"])
        self.assertIn("[cancelled]", res.output)
        # cancelled turn is dropped from history
        self.assertEqual([m["content"] for m in bodies[1]["messages"]], ["second"])


if __name__ == "__main__":
    unittest.main()
