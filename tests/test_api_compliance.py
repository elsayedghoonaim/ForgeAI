from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx

# Ensure the src folder is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.api.server import create_app
from forgeai.core.backends.factory import create_backend, resolve_backend
from forgeai.core.config import BackendType, DevToolSettings, QuantizationType


class FakeStreamingEngine:
    def __init__(self) -> None:
        self.is_running = True
        self.supports_streaming = True
        self.settings = SimpleNamespace(model_name="test-model")

    def build_prompt(self, messages):
        return "<templated-prompt>"

    async def generate(self, prompt, max_tokens, temperature, top_p, stop):
        return SimpleNamespace(
            text="hello from forgeai",
            prompt_tokens=4,
            completion_tokens=3,
            total_tokens=7,
            finish_reason="stop",
        )

    async def generate_stream(self, prompt, max_tokens, temperature, top_p, stop):
        yield "hello"
        yield " world"


class ApiComplianceTests(unittest.IsolatedAsyncioTestCase):
    def async_client(self, app):
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    def test_backend_vllm_only_contract(self) -> None:
        """Verify that resolve_backend unambiguously returns VLLM and never llama.cpp."""
        settings = DevToolSettings(model_name="meta-llama/Llama-3-8B-Instruct")
        backend_type = resolve_backend(settings)
        self.assertEqual(backend_type, BackendType.VLLM)
        self.assertEqual(backend_type.value, "vllm")

    def test_gguf_model_path_rejection(self) -> None:
        """Verify that .gguf model paths/references are explicitly rejected with clear error text."""
        # 1. DevToolSettings construction rejection
        with self.assertRaises(ValueError) as ctx:
            DevToolSettings(model_name="model.gguf")
        self.assertIn("GGUF model format is unsupported in ForgeAI v2.0+", str(ctx.exception))
        self.assertIn("llama.cpp has been removed in favor of vLLM", str(ctx.exception))

        # 2. Factory boundary rejection using validation-bypass object
        bypassed_settings = DevToolSettings.model_construct(model_name="model.gguf")
        with self.assertRaises(ValueError) as ctx_resolve:
            resolve_backend(bypassed_settings)
        self.assertIn("GGUF model format is unsupported in ForgeAI v2.0+", str(ctx_resolve.exception))

        with self.assertRaises(ValueError) as ctx_create:
            create_backend(bypassed_settings)
        self.assertIn("GGUF model format is unsupported in ForgeAI v2.0+", str(ctx_create.exception))

    def test_legacy_llamacpp_options_and_backend_rejection(self) -> None:
        """Verify that removed llama.cpp options and legacy backends are rejected with actionable errors."""
        # 1. backend="llama_cpp"
        with self.assertRaises(ValueError) as ctx_backend:
            DevToolSettings(backend="llama_cpp")  # type: ignore[arg-type]
        self.assertIn("llama.cpp backend and legacy options have been removed", str(ctx_backend.exception))
        self.assertIn("vLLM is now the sole inference engine", str(ctx_backend.exception))

        # 2. Removed field n_ctx
        with self.assertRaises(ValueError) as ctx_nctx:
            DevToolSettings(n_ctx=4096)
        self.assertIn("llama.cpp backend and legacy options have been removed", str(ctx_nctx.exception))

        # 3. Removed field n_gpu_layers
        with self.assertRaises(ValueError) as ctx_ngpu:
            DevToolSettings(n_gpu_layers=35)
        self.assertIn("llama.cpp backend and legacy options have been removed", str(ctx_ngpu.exception))

        # 4. Accepted backend="vllm"
        valid_settings = DevToolSettings(model_name="meta-llama/Llama-3-8B", backend=BackendType.VLLM)
        self.assertEqual(valid_settings.backend, BackendType.VLLM)

    def test_legacy_quantization_gguf_removed(self) -> None:
        """Verify that GGUF quantization variant is removed and not accepted."""
        allowed_quant = [e.value for e in QuantizationType]
        self.assertNotIn("gguf", allowed_quant)
        with self.assertRaises(ValueError) as ctx_quant:
            DevToolSettings(quantization="gguf")  # type: ignore[arg-type]
        self.assertIn("GGUF model format and quantization variants are unsupported", str(ctx_quant.exception))

    async def test_openai_api_streaming_compliance(self) -> None:
        """Assert FastAPI routes conform to standard OpenAI specifications for streaming completions."""
        engine = FakeStreamingEngine()
        app = create_app(engine=engine)

        async with self.async_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True
                }
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")

            lines = [line.strip() for line in response.text.split("\n") if line.strip()]
            self.assertTrue(len(lines) >= 3)

            # Verify SSE event format (data: json)
            first_chunk = lines[0]
            self.assertTrue(first_chunk.startswith("data: "))
            data_json = json.loads(first_chunk[6:])
            self.assertEqual(data_json["object"], "chat.completion.chunk")
            self.assertEqual(data_json["choices"][0]["delta"]["content"], "hello")

            second_chunk = lines[1]
            self.assertTrue(second_chunk.startswith("data: "))
            data_json_2 = json.loads(second_chunk[6:])
            self.assertEqual(data_json_2["choices"][0]["delta"]["content"], " world")

            # Verify termination chunk
            termination = lines[-1]
            self.assertEqual(termination, "data: [DONE]")


if __name__ == "__main__":
    unittest.main()
