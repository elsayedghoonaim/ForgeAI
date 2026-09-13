from __future__ import annotations

import asyncio
import sys
import unittest
from inspect import iscoroutinefunction
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the src folder is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.cli.runtime import (
    recommend_chat_max_model_len,
    recommend_chat_max_num_seqs,
    recommend_run_max_model_len,
)
from forgeai.core.backends.base import BaseBackend, GenerationResult
from forgeai.core.backends.vllm_backend import VLLMBackend
from forgeai.core.config import DevToolSettings
from forgeai.core.engine import DevToolEngine
from forgeai.utils.gpu import GPUInfo, GPUTopology


class DummyBackend(BaseBackend):
    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        return GenerationResult(text="dummy")

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        top_k: int | None = None,
    ):
        yield "dummy"

    def initialize(self) -> None:
        pass

    def build_prompt(self, messages: list[dict[str, str]]) -> str:
        return "dummy"

    def shutdown(self) -> None:
        pass

    @property
    def supports_streaming(self) -> bool:
        return True


class ArchitecturePatchesTests(unittest.TestCase):
    def test_async_generation_signature(self) -> None:
        """Verify that BaseBackend, DevToolEngine, and backends declare async generate methods."""
        # 1. BaseBackend
        self.assertTrue(iscoroutinefunction(BaseBackend.generate))

        # 2. Dummy subclass
        self.assertTrue(iscoroutinefunction(DummyBackend.generate))

        # 3. VLLMBackend
        self.assertTrue(iscoroutinefunction(VLLMBackend.generate))

        # 4. DevToolEngine
        self.assertTrue(iscoroutinefunction(DevToolEngine.generate))

    @patch("forgeai.core.backends.vllm_backend.console")
    def test_vllm_backend_shutdown_reclaim(self, mock_console) -> None:
        """Verify that VLLMBackend.shutdown terminates Ray and frees CUDA caches."""
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = True

        mock_torch = MagicMock()

        settings = DevToolSettings(model_name="dummy-model")
        backend = VLLMBackend(settings)
        backend._engine = MagicMock()
        backend._is_running = True

        with (
            patch.dict("sys.modules", {"ray": mock_ray, "torch": mock_torch, "gc": MagicMock()}),
        ):
            backend.shutdown()

        # Check Ray was shut down
        mock_ray.shutdown.assert_called_once()
        # Check PyTorch CUDA cache emptied
        mock_torch.cuda.empty_cache.assert_called_once()
        # Check engine is set to None and is_running is False
        self.assertFalse(backend.is_running)
        self.assertIsNone(backend._engine)

    def test_autotuning_model_weight_subtraction(self) -> None:
        """Verify autotuning VRAM recommendation correctly estimates model footprint and clamps it."""
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="Test GPU",
                    total_memory_mb=12288,
                    free_memory_mb=10000,
                )
            ],
            recommended_tp_size=1,
        )

        mock_est = MagicMock()
        mock_est.model_params_mb = 4000.0
        mock_est.activation_mb = 1000.0
        mock_est.overhead_mb = 500.0

        with (
            patch("forgeai.cli.runtime.detect_gpus", return_value=topology),
            patch("forgeai.utils.memory_estimator.estimate_from_preset", return_value=mock_est)
        ):
            seqs = recommend_chat_max_num_seqs(topology, tensor_parallel_size=1, model_name="llama-3-8b")
            model_len = recommend_chat_max_model_len(topology, tensor_parallel_size=1, model_name="llama-3-8b")
            run_model_len = recommend_run_max_model_len(topology, tensor_parallel_size=1, model_name="llama-3-8b")

        self.assertEqual(seqs, 4)
        self.assertEqual(model_len, 4096)
        self.assertEqual(run_model_len, 8192)

    def test_autotuning_clamping(self) -> None:
        """Verify autotuning clamps adjusted VRAM to minimum of 512MB to prevent negative numbers."""
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="Test GPU",
                    total_memory_mb=12288,
                    free_memory_mb=1000,
                )
            ],
            recommended_tp_size=1,
        )

        mock_est = MagicMock()
        mock_est.model_params_mb = 7000.0
        mock_est.activation_mb = 500.0
        mock_est.overhead_mb = 500.0

        with (
            patch("forgeai.cli.runtime.detect_gpus", return_value=topology),
            patch("forgeai.utils.memory_estimator.estimate_from_preset", return_value=mock_est)
        ):
            seqs = recommend_chat_max_num_seqs(topology, tensor_parallel_size=1, model_name="large-model")
            model_len = recommend_chat_max_model_len(topology, tensor_parallel_size=1, model_name="large-model")

        self.assertEqual(seqs, 1)
        self.assertEqual(model_len, 2048)

    def test_devtool_engine_lock_serialization(self) -> None:
        """Verify that DevToolEngine locks generation requests."""
        settings = DevToolSettings(model_name="dummy-model")
        engine = DevToolEngine(settings)
        engine._backend = DummyBackend(settings)
        engine._backend._is_running = True

        async def run_test():
            self.assertFalse(engine._lock.locked())
            async with engine._lock:
                self.assertTrue(engine._lock.locked())

            original_generate = engine._backend.generate

            async def mock_generate(*args, **kwargs):
                self.assertTrue(engine._lock.locked())
                return await original_generate(*args, **kwargs)

            engine._backend.generate = mock_generate
            result = await engine.generate("hello")
            self.assertEqual(result.text, "dummy")
            self.assertFalse(engine._lock.locked())

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
