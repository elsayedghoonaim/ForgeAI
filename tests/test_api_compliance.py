from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

# Ensure the src folder is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.api.server import create_app
from forgeai.core.backends.llamacpp_backend import LlamaCppBackend
from forgeai.core.config import DevToolSettings


class MockLlamaEngine:
    def __init__(self, metadata=None, bos_token_id=1, eos_token_id=2):
        self.metadata = metadata or {}
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id

    def token_bos(self):
        return self._bos_token_id

    def token_eos(self):
        return self._eos_token_id

    def detokenize(self, tokens):
        if tokens == [self._bos_token_id]:
            return b"<s>"
        if tokens == [self._eos_token_id]:
            return b"</s>"
        return b""


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

    def test_llamacpp_jinja_rendering(self) -> None:
        """Verify that LlamaCppBackend renders Jinja templates correctly when present in metadata."""
        settings = DevToolSettings(model_name="custom-model")
        backend = LlamaCppBackend(settings)
        
        # Set up engine with metadata Jinja template
        jinja_template = "{% for msg in messages %}<|role|>-{{ msg.role }}\n<|content|>-{{ msg.content }}\n{% endfor %}"
        mock_engine = MockLlamaEngine(metadata={"tokenizer.chat_template": jinja_template})
        backend._engine = mock_engine
        
        messages = [
            {"role": "system", "content": "You are a test helper."},
            {"role": "user", "content": "Hello!"}
        ]
        
        prompt = backend.build_prompt(messages)
        expected = "<|role|>-system\n<|content|>-You are a test helper.\n<|role|>-user\n<|content|>-Hello!\n"
        self.assertEqual(prompt, expected)

    def test_llamacpp_jinja_bos_eos_rendering(self) -> None:
        """Verify that LlamaCppBackend passes detokenized bos_token and eos_token to Jinja rendering."""
        settings = DevToolSettings(model_name="custom-model")
        backend = LlamaCppBackend(settings)
        
        # Template using bos/eos
        jinja_template = "{{ bos_token }}{% for msg in messages %}{{ msg.content }}{% endfor %}{{ eos_token }}"
        mock_engine = MockLlamaEngine(metadata={"tokenizer.chat_template": jinja_template})
        backend._engine = mock_engine
        
        messages = [{"role": "user", "content": "content"}]
        prompt = backend.build_prompt(messages)
        self.assertEqual(prompt, "<s>content</s>")

    def test_llamacpp_llama3_fallback(self) -> None:
        """Verify dynamic fallback to Llama-3 instruction template structure based on model name."""
        settings = DevToolSettings(model_name="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf")
        backend = LlamaCppBackend(settings)
        backend._engine = None  # No engine, forcing fallback
        
        messages = [
            {"role": "system", "content": "Sys prompt"},
            {"role": "user", "content": "Hello"}
        ]
        prompt = backend.build_prompt(messages)
        self.assertIn("<|begin_of_text|>", prompt)
        self.assertIn("<|start_header_id|>system<|end_header_id|>\n\nSys prompt<|eot_id|>", prompt)
        self.assertIn("<|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|>", prompt)
        self.assertIn("<|start_header_id|>assistant<|end_header_id|>\n\n", prompt)

    def test_llamacpp_chatml_fallback(self) -> None:
        """Verify dynamic fallback to ChatML templates based on model name."""
        settings = DevToolSettings(model_name="qwen2-7b-instruct.gguf")
        backend = LlamaCppBackend(settings)
        backend._engine = None
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        prompt = backend.build_prompt(messages)
        self.assertEqual(prompt, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n")

    def test_llamacpp_llama2_fallback(self) -> None:
        """Verify dynamic fallback to Llama-2/Mistral tags based on model name."""
        settings = DevToolSettings(model_name="mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        backend = LlamaCppBackend(settings)
        backend._engine = None
        
        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "End"}
        ]
        prompt = backend.build_prompt(messages)
        self.assertEqual(prompt, "<s>[INST] <<SYS>>\nSys\n<</SYS>>\n\nHello [/INST] Hi </s><s>[INST] End [/INST]")

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
