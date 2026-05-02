from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.cli.commands.run import run
from forgeai.cli.runtime import RuntimeTuning


class FakeEngine:
    init_kwargs: dict[str, object] = {}
    stream_called = False
    shutdown_called = False

    def __init__(self, settings, *, streaming: bool = False, quiet_startup: bool = False) -> None:
        FakeEngine.init_kwargs = {
            "settings": settings,
            "streaming": streaming,
            "quiet_startup": quiet_startup,
        }
        FakeEngine.stream_called = False
        FakeEngine.shutdown_called = False
        self.last_result = None

    def initialize(self) -> None:
        return None

    async def generate_stream(self, *, prompt, max_tokens, temperature, top_p):
        del prompt, max_tokens, temperature, top_p
        FakeEngine.stream_called = True
        yield "Hel"
        yield "lo"
        self.last_result = SimpleNamespace(
            total_tokens=5,
            tokens_per_second=2.5,
            elapsed_seconds=2.0,
        )

    def shutdown(self) -> None:
        FakeEngine.shutdown_called = True


class RunCommandTests(unittest.TestCase):
    def test_run_streams_by_default(self) -> None:
        tuning = RuntimeTuning(
            profile="run",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.80,
            max_num_seqs=1,
            max_model_len=4096,
            max_num_batched_tokens=512,
            enforce_eager=True,
        )

        with (
            patch("forgeai.cli.runtime.resolve_runtime_tuning", return_value=tuning),
            patch("forgeai.cli.runtime.print_runtime_tuning"),
            patch("forgeai.core.telemetry.track_event"),
            patch("forgeai.models.zoo.resolve_model_name", return_value="google/gemma-4-E2B-it"),
            patch("forgeai.core.engine.DevToolEngine", FakeEngine),
        ):
            run(
                model="google/gemma-4-E2B-it",
                prompt="Hello",
                max_tokens=32,
                temperature=0.7,
                top_p=0.95,
                auto_optimize=False,
                dry_run=False,
                backend="auto",
                n_gpu_layers=0,
                n_ctx=4096,
                gpu_utilization=None,
                tensor_parallel=None,
                stream=True,
                startup_logs=True,
            )

        self.assertTrue(FakeEngine.init_kwargs["streaming"])
        self.assertFalse(FakeEngine.init_kwargs["quiet_startup"])
        self.assertTrue(FakeEngine.stream_called)
        self.assertTrue(FakeEngine.shutdown_called)


if __name__ == "__main__":
    unittest.main()
