"""
Deterministic unit tests for Ollama-style daemon client and CLI commands.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
from typer.testing import CliRunner

from forgeai.cli.main import app
from forgeai.cli.runtime import DaemonClient, DaemonClientError, resolve_base_url
from forgeai.models.manifest import ForgeAIManifest

runner = CliRunner(mix_stderr=False)


class FakeResponse:
    def __init__(self, status_code: int = 200, lines: list[str] | None = None, content: bytes = b"", error_json: dict | None = None) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self._content = content
        self._error_json = error_json
        self.is_closed = False

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.is_closed = True

    def read(self) -> bytes:
        if self._error_json is not None:
            return json.dumps(self._error_json).encode("utf-8")
        return self._content

    def json(self) -> Any:
        if self._content == b"not json":
            raise json.JSONDecodeError("Expecting value", "not json", 0)
        if self._error_json is not None:
            return self._error_json
        return json.loads(self._content.decode("utf-8"))

    @property
    def text(self) -> str:
        return self._content.decode("utf-8")

    def iter_lines(self) -> Any:
        for line in self._lines:
            yield line


class FakeClient:
    def __init__(self, response: FakeResponse | None = None, stream_exc: Exception | None = None, req_exc: Exception | None = None, timeout: Any = None) -> None:
        self.response = response or FakeResponse()
        self.stream_exc = stream_exc
        self.req_exc = req_exc
        self.is_closed = False

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.is_closed = True

    def request(self, method: str, url: str, json: Any = None, params: Any = None) -> FakeResponse:
        if self.req_exc:
            raise self.req_exc
        return self.response

    def stream(self, method: str, url: str, json: Any = None, params: Any = None) -> FakeResponse:
        if self.stream_exc:
            raise self.stream_exc
        return self.response


class BaseUrlAndDaemonClientTests(unittest.TestCase):
    def test_base_url_resolution_defaults_and_overrides(self) -> None:
        # Default
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_base_url(), "http://127.0.0.1:11434")

        # Explicit host and port
        self.assertEqual(
            resolve_base_url(host="localhost", port=9000), "http://localhost:9000"
        )

        # Base URL parameter
        self.assertEqual(
            resolve_base_url(base_url="http://custom-host:8080/"),
            "http://custom-host:8080",
        )
        self.assertEqual(
            resolve_base_url(base_url="custom-host:8080"),
            "http://custom-host:8080",
        )

        # Environment variables FORGEAI_HOST / FORGEAI_PORT
        with patch.dict(
            os.environ, {"FORGEAI_HOST": "10.0.0.1", "FORGEAI_PORT": "1234"}
        ):
            self.assertEqual(resolve_base_url(), "http://10.0.0.1:1234")

        # Environment variables forgeai_host / forgeai_port
        with patch.dict(
            os.environ, {"forgeai_host": "10.0.0.2", "forgeai_port": "5678"}
        ):
            self.assertEqual(resolve_base_url(), "http://10.0.0.2:5678")

    def test_base_url_resolution_invalid_and_out_of_range_ports(self) -> None:
        with self.assertRaises(DaemonClientError) as ctx1:
            resolve_base_url(port="invalid")
        self.assertIn("Invalid port number", str(ctx1.exception))

        with self.assertRaises(DaemonClientError) as ctx2:
            resolve_base_url(port=70000)
        self.assertIn("out of valid range", str(ctx2.exception))

        with self.assertRaises(DaemonClientError) as ctx3:
            resolve_base_url(port=0)
        self.assertIn("out of valid range", str(ctx3.exception))

        with patch.dict(os.environ, {"FORGEAI_PORT": "not_a_number"}):
            with self.assertRaises(DaemonClientError) as ctx4:
                resolve_base_url()
            self.assertIn("Invalid port number", str(ctx4.exception))

    def test_base_url_resolution_host_with_scheme_and_port(self) -> None:
        # Retains 8080 even if FORGEAI_PORT is set in env
        with patch.dict(os.environ, {"FORGEAI_PORT": "9999"}):
            self.assertEqual(
                resolve_base_url(host="localhost:8080"), "http://localhost:8080"
            )
            self.assertEqual(
                resolve_base_url(host="http://localhost:8080"), "http://localhost:8080"
            )
            self.assertEqual(
                resolve_base_url(host="https://localhost:8080"), "https://localhost:8080"
            )

        # Explicit CLI --port option still overrides host-embedded port
        with patch.dict(os.environ, {"FORGEAI_PORT": "9999"}):
            self.assertEqual(
                resolve_base_url(host="http://localhost:8080", port=5555),
                "http://localhost:5555",
            )

        # If host lacks a port, explicit port then environment port then 11434 applies
        with patch.dict(os.environ, {"FORGEAI_PORT": "9999"}):
            self.assertEqual(
                resolve_base_url(host="localhost", port=5555),
                "http://localhost:5555",
            )
            self.assertEqual(
                resolve_base_url(host="localhost"),
                "http://localhost:9999",
            )

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                resolve_base_url(host="localhost"),
                "http://localhost:11434",
            )

    def test_base_url_resolution_cli_overrides_env(self) -> None:
        with patch.dict(
            os.environ, {"FORGEAI_HOST": "10.0.0.1", "FORGEAI_PORT": "1234"}
        ):
            self.assertEqual(
                resolve_base_url(host="192.168.1.1", port=5555),
                "http://192.168.1.1:5555",
            )

    def test_base_url_resolution_trailing_slash_normalization(self) -> None:
        self.assertEqual(
            resolve_base_url(host="http://127.0.0.1:11434/"), "http://127.0.0.1:11434"
        )
        self.assertEqual(
            resolve_base_url(base_url="http://127.0.0.1:11434///"), "http://127.0.0.1:11434"
        )

    def test_daemon_client_json_request_success(self) -> None:
        fake_resp = FakeResponse(status_code=200, content=json.dumps({"status": "ok"}).encode("utf-8"))
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            res = client.request("GET", "/api/tags")

        self.assertEqual(res, {"status": "ok"})
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_request_malformed_json_closes_client(self) -> None:
        fake_resp = FakeResponse(status_code=200, content=b"not json")
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            with self.assertRaises(DaemonClientError) as ctx:
                client.request("GET", "/api/tags")

        self.assertIn("Malformed JSON response", str(ctx.exception))
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_stream_exhaustion_closes_resources(self) -> None:
        fake_resp = FakeResponse(
            lines=[json.dumps({"status": "pulling"}), json.dumps({"status": "success"})]
        )
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            items = list(client.stream("POST", "/api/pull", json_data={"name": "m"}))

        self.assertEqual(len(items), 2)
        self.assertTrue(fake_resp.is_closed)
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_stream_early_generator_close(self) -> None:
        fake_resp = FakeResponse(
            lines=[json.dumps({"status": "chunk1"}), json.dumps({"status": "chunk2"})]
        )
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            gen = client.stream("POST", "/api/pull")
            item1 = next(gen)
            self.assertEqual(item1["status"], "chunk1")
            gen.close()

        self.assertTrue(fake_resp.is_closed)
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_stream_mid_stream_error_closes_resources(self) -> None:
        fake_resp = FakeResponse(
            lines=[json.dumps({"status": "ok"}), json.dumps({"error": "network failure mid-stream"})]
        )
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            gen = client.stream("POST", "/api/pull")
            self.assertEqual(next(gen)["status"], "ok")
            with self.assertRaises(DaemonClientError) as ctx:
                next(gen)

        self.assertIn("network failure mid-stream", str(ctx.exception))
        self.assertTrue(fake_resp.is_closed)
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_stream_malformed_ndjson_closes_resources(self) -> None:
        fake_resp = FakeResponse(lines=["{bad json line"])
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            gen = client.stream("POST", "/api/pull")
            with self.assertRaises(DaemonClientError) as ctx:
                next(gen)

        self.assertIn("Malformed NDJSON chunk", str(ctx.exception))
        self.assertTrue(fake_resp.is_closed)
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_stream_non_2xx_error_closes_resources(self) -> None:
        fake_resp = FakeResponse(status_code=400, error_json={"error": "Ollama request failed"})
        fake_client = FakeClient(response=fake_resp)

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            gen = client.stream("POST", "/api/pull")
            with self.assertRaises(DaemonClientError) as ctx:
                next(gen)

        self.assertIn("Ollama request failed", str(ctx.exception))
        self.assertTrue(fake_resp.is_closed)
        self.assertTrue(fake_client.is_closed)

    def test_daemon_client_stream_connection_failure_closes_client(self) -> None:
        fake_client = FakeClient(stream_exc=httpx.ConnectError("Conn failed"))

        with patch("httpx.Client", return_value=fake_client):
            client = DaemonClient(base_url="http://127.0.0.1:11434")
            gen = client.stream("POST", "/api/pull")
            with self.assertRaises(DaemonClientError) as ctx:
                next(gen)

        self.assertIn("Could not connect to daemon", str(ctx.exception))
        self.assertTrue(fake_client.is_closed)


class OllamaCliCommandsTests(unittest.TestCase):
    def test_cli_registration_and_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("ls", result.stdout)
        self.assertIn("ps", result.stdout)
        self.assertIn("show", result.stdout)
        self.assertIn("rm", result.stdout)
        self.assertIn("stop", result.stdout)
        self.assertIn("create", result.stdout)

        for cmd in ["ls", "ps", "show", "rm", "stop", "create"]:
            res = runner.invoke(app, [cmd, "--help"])
            self.assertEqual(res.exit_code, 0)

    @patch.object(DaemonClient, "request")
    def test_ls_command_success_and_empty(self, mock_request: MagicMock) -> None:
        # Success with models
        mock_request.return_value = {
            "models": [
                {
                    "name": "gemma-2-9b-it:latest",
                    "size": 10737418240,
                    "modified_at": "2026-08-09T10:00:00Z",
                    "details": {
                        "format": "safetensors",
                        "weight_quantization": "none",
                        "kv_cache_dtype": "auto",
                    },
                }
            ]
        }
        result = runner.invoke(app, ["ls"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("gemma-2-9b-it:latest", result.stdout)
        self.assertIn("10.0 GB", result.stdout)
        self.assertIn("safetensors", result.stdout)
        mock_request.assert_called_once_with("GET", "/api/tags")

        # Empty list
        mock_request.reset_mock()
        mock_request.return_value = {"models": []}
        result_empty = runner.invoke(app, ["ls"])
        self.assertEqual(result_empty.exit_code, 0)
        self.assertIn("No models registered", result_empty.stdout)

    @patch.object(DaemonClient, "request")
    def test_ps_command_success_and_empty(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {
            "models": [
                {
                    "name": "gemma-2-9b-it:latest",
                    "expires_at": "2100-01-01T00:00:00Z",
                    "size_vram": 0,
                    "details": {
                        "format": "safetensors",
                        "weight_quantization": "none",
                        "kv_cache_dtype": "auto",
                    },
                }
            ]
        }
        result = runner.invoke(app, ["ps"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("gemma-2-9b-it:latest", result.stdout)
        self.assertIn("Unknown", result.stdout)
        mock_request.assert_called_once_with("GET", "/api/ps")

        mock_request.reset_mock()
        mock_request.return_value = {"models": []}
        result_empty = runner.invoke(app, ["ps"])
        self.assertEqual(result_empty.exit_code, 0)
        self.assertIn("No active models running", result_empty.stdout)

    @patch.object(DaemonClient, "request")
    def test_show_command_table_and_json(self, mock_request: MagicMock) -> None:
        mock_data = {
            "modelfile": "# ForgeAI Local Manifest\nname: gemma-2-9b-it:latest",
            "parameters": "temperature 0.7",
            "template": "jinja",
            "system": "You are a helpful assistant.",
            "details": {"format": "safetensors", "weight_quantization": "none"},
            "model_info": {"general.architecture": "vLLM"},
        }
        mock_request.return_value = mock_data

        # Text display
        res_text = runner.invoke(app, ["show", "gemma-2-9b-it:latest"])
        self.assertEqual(res_text.exit_code, 0)
        self.assertIn("gemma-2-9b-it:latest", res_text.stdout)
        self.assertIn("temperature 0.7", res_text.stdout)
        mock_request.assert_called_with(
            "POST", "/api/show", json_data={"name": "gemma-2-9b-it:latest"}
        )

        # JSON display
        res_json = runner.invoke(app, ["show", "gemma-2-9b-it:latest", "--json"])
        self.assertEqual(res_json.exit_code, 0)
        self.assertIn('"modelfile"', res_json.stdout)

    @patch.object(DaemonClient, "request")
    def test_rm_command_success(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {}
        result = runner.invoke(app, ["rm", "gemma-2-9b-it:latest"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("deleted 'gemma-2-9b-it:latest'", result.stdout)
        mock_request.assert_called_once_with(
            "DELETE", "/api/delete", json_data={"name": "gemma-2-9b-it:latest"}
        )

    @patch.object(DaemonClient, "request")
    def test_stop_command_success(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"done": True}
        result = runner.invoke(app, ["stop", "gemma-2-9b-it:latest"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("stopped 'gemma-2-9b-it:latest'", result.stdout)
        mock_request.assert_called_once_with(
            "POST",
            "/api/generate",
            json_data={
                "model": "gemma-2-9b-it:latest",
                "prompt": "",
                "stream": False,
                "keep_alive": 0,
            },
        )

    @patch("forgeai.models.registry.ModelRegistry.register_manifest")
    def test_create_command_valid_yaml(self, mock_register: MagicMock) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(
                "model: google/gemma-2-9b-it\nsource_kind: huggingface\nrevision: main\n"
            )
            tmp_path = Path(f.name)

        try:
            res = runner.invoke(
                app, ["create", "my-gemma:v1", "-f", str(tmp_path)]
            )
            self.assertEqual(res.exit_code, 0)
            self.assertIn("Created model 'my-gemma:v1'", res.stdout)

            mock_register.assert_called_once()
            manifest_arg = mock_register.call_args[0][0]
            self.assertIsInstance(manifest_arg, ForgeAIManifest)
            self.assertEqual(manifest_arg.name, "my-gemma:v1")
            self.assertEqual(manifest_arg.model, "google/gemma-2-9b-it")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_create_command_rejects_ollama_modelfile(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
            f.write("FROM llama3\nPARAMETER temperature 0.7\nSYSTEM You are helpful.")
            tmp_path = Path(f.name)

        try:
            res = runner.invoke(
                app, ["create", "my-model:latest", "-f", str(tmp_path)]
            )
            self.assertEqual(res.exit_code, 1)
            self.assertIn("Modelfile text is unsupported", res.stderr)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_create_command_rejects_invalid_yaml_parser_error(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("foo: [unclosed list")
            tmp_path = Path(f.name)

        try:
            res = runner.invoke(
                app, ["create", "my-model:latest", "-f", str(tmp_path)]
            )
            self.assertEqual(res.exit_code, 1)
            self.assertIn("Invalid YAML content in manifest file", res.stderr)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_create_command_rejects_non_mapping_yaml(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write('"just a plain string content"')
            tmp_path = Path(f.name)

        try:
            res = runner.invoke(
                app, ["create", "my-model:latest", "-f", str(tmp_path)]
            )
            self.assertEqual(res.exit_code, 1)
            self.assertIn("must contain a YAML dictionary", res.stderr)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_create_command_rejects_gguf_manifest(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write("model: model.GGUF\nsource_kind: huggingface\n")
            tmp_path = Path(f.name)

        try:
            res = runner.invoke(
                app, ["create", "my-gguf:latest", "-f", str(tmp_path)]
            )
            self.assertEqual(res.exit_code, 1)
            self.assertIn("GGUF model format is unsupported", res.stderr)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @patch.object(DaemonClient, "request")
    def test_connection_failure_cli_error_handling(
        self, mock_request: MagicMock
    ) -> None:
        mock_request.side_effect = DaemonClientError(
            "Could not connect to daemon at http://127.0.0.1:11434. Is the server running? Start it with: forgeai serve"
        )
        res = runner.invoke(app, ["ls"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Could not connect to daemon", res.stderr)
        self.assertNotIn("Traceback (most recent call last):", res.stderr)

    def test_invalid_cli_port_error_handling(self) -> None:
        res = runner.invoke(app, ["ls", "--port", "70000"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("out of valid range", res.stderr)
        self.assertNotIn("Traceback (most recent call last):", res.stderr)

    def test_invalid_env_port_error_handling(self) -> None:
        with patch.dict(os.environ, {"FORGEAI_PORT": "invalid_port"}):
            res = runner.invoke(app, ["ls"])
            self.assertEqual(res.exit_code, 1)
            self.assertIn("Invalid port number", res.stderr)
            self.assertNotIn("Traceback (most recent call last):", res.stderr)


class OllamaServePullRunCommandsTests(unittest.TestCase):
    @patch("uvicorn.run")
    @patch("forgeai.api.server.create_app")
    @patch("forgeai.api.runtime.SharedRuntimeAdapter")
    @patch("forgeai.core.engine.EngineManager")
    @patch("forgeai.models.loader.CacheManager")
    @patch("forgeai.models.registry.ModelRegistry")
    @patch("forgeai.security.compliance.audit_logger.AuditLogger")
    @patch("forgeai.security.rate_limit.MemoryRateLimiter")
    def test_serve_defaults_and_wiring(
        self,
        mock_rate: MagicMock,
        mock_audit: MagicMock,
        mock_registry: MagicMock,
        mock_cache: MagicMock,
        mock_engine_mgr: MagicMock,
        mock_adapter_cls: MagicMock,
        mock_create_app: MagicMock,
        mock_uvicorn: MagicMock,
    ) -> None:
        mock_create_app.return_value = MagicMock()
        mock_adapter_inst = MagicMock()
        mock_adapter_cls.return_value = mock_adapter_inst

        res = runner.invoke(app, ["serve"])
        self.assertEqual(res.exit_code, 0)

        # Assert exactly one of each component created with parameterless canonical CacheManager
        mock_registry.assert_called_once_with()
        mock_cache.assert_called_once_with()
        mock_engine_mgr.assert_called_once()
        mock_adapter_cls.assert_called_once_with(
            engine_manager=mock_engine_mgr.return_value,
            model_registry=mock_registry.return_value,
            cache_manager=mock_cache.return_value,
            settings=mock_engine_mgr.call_args.kwargs["settings"],
        )

        mock_create_app.assert_called_once()
        self.assertEqual(
            mock_create_app.call_args.kwargs["runtime_adapter"], mock_adapter_inst
        )

        mock_uvicorn.assert_called_once_with(
            mock_create_app.return_value,
            host="127.0.0.1",
            port=11434,
            workers=1,
            log_level="info",
        )

    def test_serve_workers_rejection(self) -> None:
        res = runner.invoke(app, ["serve", "--workers", "2"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("workers must be set to 1", res.stderr)

    def test_serve_configuration_failure_no_traceback(self) -> None:
        for bad_args in [
            ["serve", "--port", "70000"],
            ["serve", "--max-loaded-models", "0"],
            ["serve", "--load-concurrency", "5", "--max-loaded-models", "1"],
            ["serve", "--request-queue-depth", "0"],
        ]:
            res = runner.invoke(app, bad_args)
            self.assertEqual(res.exit_code, 1)
            self.assertIn("Error:", res.stderr)
            self.assertNotIn("Traceback (most recent call last):", res.stderr)

    def test_serve_legacy_option_rejection(self) -> None:
        for bad_args in [
            ["serve", "gemma-2-9b-it"],
            ["serve", "--backend", "vllm"],
            ["serve", "--n-gpu-layers", "10"],
            ["serve", "--n-ctx", "2048"],
            ["serve", "--auto-optimize"],
        ]:
            res = runner.invoke(app, bad_args)
            self.assertNotEqual(res.exit_code, 0)

    @patch.object(DaemonClient, "stream")
    def test_pull_streaming_success(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = iter([
            {"status": "pulling manifest"},
            {"status": "downloading weights"},
            {"status": "success", "digest": "sha256:abc"},
        ])
        res = runner.invoke(app, ["pull", "gemma-2-9b-it"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("status: pulling manifest", res.stdout)
        self.assertIn("status: success digest: sha256:abc", res.stdout)
        self.assertIn("Model ready", res.stdout)
        mock_stream.assert_called_once_with(
            "POST",
            "/api/pull",
            json_data={"name": "gemma-2-9b-it", "stream": True},
        )

    @patch.object(DaemonClient, "stream")
    def test_pull_empty_or_premature_stream_fails(self, mock_stream: MagicMock) -> None:
        # Empty stream
        mock_stream.return_value = iter([])
        res = runner.invoke(app, ["pull", "gemma-2-9b-it"])
        self.assertEqual(res.exit_code, 1)
        self.assertNotIn("Model ready", res.stdout)
        self.assertIn("Error:", res.stderr)

        # Stream ending without success event
        mock_stream.return_value = iter([{"status": "pulling manifest"}])
        res2 = runner.invoke(app, ["pull", "gemma-2-9b-it"])
        self.assertEqual(res2.exit_code, 1)
        self.assertNotIn("Model ready", res2.stdout)
        self.assertIn("Error:", res2.stderr)

    @patch.object(DaemonClient, "request")
    def test_pull_nonsuccess_nonstream_fails(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"status": "failed", "error": "Repo not found"}
        res = runner.invoke(app, ["pull", "gemma-2-9b-it", "--no-stream"])
        self.assertEqual(res.exit_code, 1)
        self.assertNotIn("Model ready", res.stdout)
        self.assertIn("Error:", res.stderr)

    @patch.object(DaemonClient, "request")
    def test_pull_no_stream(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"status": "success", "digest": "sha256:abc"}
        res = runner.invoke(app, ["pull", "gemma-2-9b-it", "--no-stream"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("status: success digest: sha256:abc", res.stdout)
        self.assertIn("Model ready", res.stdout)
        mock_request.assert_called_once_with(
            "POST",
            "/api/pull",
            json_data={"name": "gemma-2-9b-it", "stream": False},
        )

    @patch.object(DaemonClient, "stream")
    def test_pull_error_behavior(self, mock_stream: MagicMock) -> None:
        mock_stream.side_effect = DaemonClientError(
            "Could not connect to daemon at http://127.0.0.1:11434. Is the server running? Start it with: forgeai serve"
        )
        res = runner.invoke(app, ["pull", "gemma-2-9b-it"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Could not connect to daemon", res.stderr)
        self.assertNotIn("Model ready", res.stdout)

    def test_pull_legacy_options_rejection(self) -> None:
        for bad_args in [
            ["pull", "m", "--token", "secret"],
            ["pull", "m", "--cache-dir", "/tmp"],
            ["pull", "m", "--revision", "main"],
            ["pull", "m", "--skip-scan"],
        ]:
            res = runner.invoke(app, bad_args)
            self.assertNotEqual(res.exit_code, 0)

    def test_invalid_pull_run_port_handled(self) -> None:
        res_pull = runner.invoke(app, ["pull", "gemma-2-9b-it", "--port", "70000"])
        self.assertEqual(res_pull.exit_code, 1)
        self.assertIn("out of valid range", res_pull.stderr)
        self.assertNotIn("Traceback (most recent call last):", res_pull.stderr)

        res_run = runner.invoke(app, ["run", "gemma-2-9b-it", "hi", "--port", "70000"])
        self.assertEqual(res_run.exit_code, 1)
        self.assertIn("out of valid range", res_run.stderr)
        self.assertNotIn("Traceback (most recent call last):", res_run.stderr)

    @patch("httpx.Client.request", side_effect=httpx.ConnectError("Failed"))
    def test_connection_text_includes_forgeai_serve(self, mock_request: MagicMock) -> None:
        res = runner.invoke(app, ["run", "gemma-2-9b-it", "hi"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Start it with: forgeai serve", res.stderr)
        self.assertNotIn("Traceback (most recent call last):", res.stderr)

    @patch.object(DaemonClient, "stream")
    def test_run_streaming_success(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = iter([
            {"model": "gemma-2-9b-it", "response": "Hello ", "done": False},
            {"model": "gemma-2-9b-it", "response": "world!", "done": False},
            {"model": "gemma-2-9b-it", "response": "", "done": True},
        ])
        res = runner.invoke(app, ["run", "gemma-2-9b-it", "Say hello"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Hello world!", res.stdout)
        mock_stream.assert_called_once_with(
            "POST",
            "/api/generate",
            json_data={"model": "gemma-2-9b-it", "prompt": "Say hello", "stream": True},
        )

    @patch.object(DaemonClient, "request")
    def test_run_no_stream(self, mock_request: MagicMock) -> None:
        mock_request.return_value = {"model": "gemma-2-9b-it", "response": "Hello world!", "done": True}
        res = runner.invoke(app, ["run", "gemma-2-9b-it", "Say hello", "--no-stream"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Hello world!", res.stdout)
        mock_request.assert_called_once_with(
            "POST",
            "/api/generate",
            json_data={"model": "gemma-2-9b-it", "prompt": "Say hello", "stream": False},
        )

    @patch.object(DaemonClient, "stream")
    def test_run_prompt_alias_and_options(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = iter([{"response": "Output"}])
        res = runner.invoke(
            app,
            [
                "run",
                "gemma-2-9b-it",
                "--prompt",
                "Alias prompt",
                "--keep-alive",
                "10m",
                "--temperature",
                "0.8",
                "--top-p",
                "0.9",
                "--top-k",
                "50",
                "--max-tokens",
                "256",
                "--stop",
                "END",
            ],
        )
        self.assertEqual(res.exit_code, 0)
        mock_stream.assert_called_once_with(
            "POST",
            "/api/generate",
            json_data={
                "model": "gemma-2-9b-it",
                "prompt": "Alias prompt",
                "stream": True,
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 50,
                    "num_predict": 256,
                    "stop": ["END"],
                },
            },
        )

    def test_run_prompt_conflict(self) -> None:
        res = runner.invoke(app, ["run", "gemma-2-9b-it", "Positional prompt", "-p", "Option prompt"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Conflicting prompt", res.stderr)

    @patch("builtins.input", side_effect=["First turn", "/bye"])
    @patch.object(DaemonClient, "stream")
    def test_run_interactive_reuse_and_exit(self, mock_stream: MagicMock, mock_input: MagicMock) -> None:
        mock_stream.return_value = iter([{"response": "Turn 1 answer"}])
        res = runner.invoke(app, ["run", "gemma-2-9b-it"])
        self.assertEqual(res.exit_code, 0)
        self.assertIn("Turn 1 answer", res.stdout)
        mock_stream.assert_called_once_with(
            "POST",
            "/api/generate",
            json_data={"model": "gemma-2-9b-it", "prompt": "First turn", "stream": True},
        )

    @patch.object(DaemonClient, "stream")
    def test_run_daemon_failure(self, mock_stream: MagicMock) -> None:
        mock_stream.side_effect = DaemonClientError(
            "Could not connect to daemon at http://127.0.0.1:11434. Is the server running? Start it with: forgeai serve"
        )
        res = runner.invoke(app, ["run", "gemma-2-9b-it", "Prompt"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("Could not connect to daemon", res.stderr)

    def test_run_legacy_flags_rejection(self) -> None:
        for bad_args in [
            ["run", "m", "p", "--backend", "vllm"],
            ["run", "m", "p", "--n-gpu-layers", "10"],
            ["run", "m", "p", "--n-ctx", "2048"],
            ["run", "m", "p", "--gpu-util", "0.8"],
            ["run", "m", "p", "--tp", "2"],
            ["run", "m", "p", "--auto-optimize"],
            ["run", "m", "p", "--dry-run"],
            ["run", "m", "p", "--startup-logs"],
        ]:
            res = runner.invoke(app, bad_args)
            self.assertNotEqual(res.exit_code, 0)

    def test_cold_startup_never_initializes_engine(self) -> None:
        with (
            patch("uvicorn.run"),
            patch("forgeai.api.server.create_app") as mock_create_app,
            patch("forgeai.api.runtime.SharedRuntimeAdapter"),
            patch("forgeai.core.engine.EngineManager"),
            patch("forgeai.models.loader.CacheManager"),
            patch("forgeai.models.registry.ModelRegistry"),
            patch("forgeai.security.compliance.audit_logger.AuditLogger"),
            patch("forgeai.security.rate_limit.MemoryRateLimiter"),
            patch("forgeai.core.engine.DevToolEngine") as mock_devtool_engine,
        ):
            mock_create_app.return_value = MagicMock()
            res = runner.invoke(app, ["serve"])
            self.assertEqual(res.exit_code, 0)
            mock_devtool_engine.assert_not_called()

    def test_no_legacy_code_invocation(self) -> None:
        with (
            patch("uvicorn.run"),
            patch("forgeai.api.server.create_app") as mock_create_app,
            patch("forgeai.core.engine.DevToolEngine") as mock_devtool_engine,
            patch("forgeai.models.loader.download_model") as mock_download_model,
            patch("forgeai.utils.gpu.detect_gpus") as mock_detect_gpus,
            patch("forgeai.models.loader.CacheManager"),
            patch("forgeai.models.registry.ModelRegistry"),
            patch("forgeai.core.engine.EngineManager"),
            patch("forgeai.api.runtime.SharedRuntimeAdapter"),
            patch("forgeai.security.compliance.audit_logger.AuditLogger"),
            patch("forgeai.security.rate_limit.MemoryRateLimiter"),
        ):
            mock_create_app.return_value = MagicMock()

            # Serve
            res_serve = runner.invoke(app, ["serve"])
            self.assertEqual(res_serve.exit_code, 0)

            # Pull (mock client request with actual success status event)
            with patch.object(
                DaemonClient,
                "stream",
                return_value=iter([{"status": "downloading"}, {"status": "success"}]),
            ):
                res_pull = runner.invoke(app, ["pull", "test-model"])
                self.assertEqual(res_pull.exit_code, 0)

            # Run (mock client request)
            with patch.object(
                DaemonClient, "stream", return_value=iter([{"response": "ok"}])
            ):
                res_run = runner.invoke(app, ["run", "test-model", "test prompt"])
                self.assertEqual(res_run.exit_code, 0)

            mock_devtool_engine.assert_not_called()
            mock_download_model.assert_not_called()
            mock_detect_gpus.assert_not_called()


if __name__ == "__main__":
    unittest.main()
