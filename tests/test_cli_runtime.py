from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.cli.runtime import resolve_runtime_tuning
from forgeai.utils.gpu import GPUInfo, GPUTopology


class RuntimeTuningTests(unittest.TestCase):
    def test_omitted_gpu_utilization_auto_tunes(self) -> None:
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="Test GPU",
                    total_memory_mb=6144,
                    free_memory_mb=5315,
                )
            ],
            recommended_tp_size=1,
        )

        with patch("forgeai.cli.runtime.detect_gpus", return_value=topology):
            tuning = resolve_runtime_tuning(
                tensor_parallel_size=None,
                gpu_memory_utilization=None,
                auto_optimize=False,
            )

        self.assertTrue(tuning.auto_gpu_utilization)
        self.assertEqual(tuning.gpu_memory_utilization, 0.80)

    def test_explicit_gpu_utilization_is_preserved(self) -> None:
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="Test GPU",
                    total_memory_mb=6144,
                    free_memory_mb=5315,
                )
            ],
            recommended_tp_size=1,
        )

        with patch("forgeai.cli.runtime.detect_gpus", return_value=topology):
            tuning = resolve_runtime_tuning(
                tensor_parallel_size=None,
                gpu_memory_utilization=0.72,
                auto_optimize=False,
            )

        self.assertFalse(tuning.auto_gpu_utilization)
        self.assertEqual(tuning.gpu_memory_utilization, 0.72)

    def test_auto_optimize_uses_recommended_tp_when_unspecified(self) -> None:
        topology = GPUTopology(
            gpus=[
                GPUInfo(index=0, name="GPU 0", total_memory_mb=12288, free_memory_mb=12000),
                GPUInfo(index=1, name="GPU 1", total_memory_mb=12288, free_memory_mb=12000),
            ],
            recommended_tp_size=2,
        )

        with patch("forgeai.cli.runtime.detect_gpus", return_value=topology):
            tuning = resolve_runtime_tuning(
                tensor_parallel_size=None,
                gpu_memory_utilization=None,
                auto_optimize=True,
            )

        self.assertTrue(tuning.auto_tensor_parallel)
        self.assertEqual(tuning.tensor_parallel_size, 2)

    def test_chat_mode_auto_tunes_chat_limits(self) -> None:
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="GPU 0",
                    total_memory_mb=6144,
                    free_memory_mb=5315,
                )
            ],
            recommended_tp_size=1,
        )

        with patch("forgeai.cli.runtime.detect_gpus", return_value=topology):
            tuning = resolve_runtime_tuning(
                tensor_parallel_size=None,
                gpu_memory_utilization=None,
                auto_optimize=False,
                chat_mode=True,
            )

        self.assertTrue(tuning.auto_max_num_seqs)
        self.assertTrue(tuning.auto_max_model_len)
        self.assertTrue(tuning.auto_enforce_eager)
        self.assertEqual(tuning.max_num_seqs, 4)
        self.assertEqual(tuning.max_model_len, 4096)
        self.assertEqual(tuning.gpu_memory_utilization, 0.80)
        self.assertTrue(tuning.enforce_eager)

    def test_chat_mode_respects_env_overrides(self) -> None:
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="GPU 0",
                    total_memory_mb=6144,
                    free_memory_mb=5315,
                )
            ],
            recommended_tp_size=1,
        )

        with (
            patch("forgeai.cli.runtime.detect_gpus", return_value=topology),
            patch.dict(
                os.environ,
                {
                    "forgeai_MAX_NUM_SEQS": "12",
                    "forgeai_MAX_MODEL_LEN": "32768",
                    "forgeai_ENFORCE_EAGER": "0",
                },
                clear=False,
            ),
        ):
            tuning = resolve_runtime_tuning(
                tensor_parallel_size=None,
                gpu_memory_utilization=None,
                auto_optimize=False,
                chat_mode=True,
            )

        self.assertFalse(tuning.auto_max_num_seqs)
        self.assertFalse(tuning.auto_max_model_len)
        self.assertFalse(tuning.auto_enforce_eager)
        self.assertIsNone(tuning.max_num_seqs)
        self.assertIsNone(tuning.max_model_len)
        self.assertFalse(tuning.enforce_eager)

    def test_run_mode_auto_tunes_one_shot_profile(self) -> None:
        topology = GPUTopology(
            gpus=[
                GPUInfo(
                    index=0,
                    name="GPU 0",
                    total_memory_mb=6144,
                    free_memory_mb=5315,
                )
            ],
            recommended_tp_size=1,
        )

        with patch("forgeai.cli.runtime.detect_gpus", return_value=topology):
            tuning = resolve_runtime_tuning(
                tensor_parallel_size=None,
                gpu_memory_utilization=None,
                auto_optimize=False,
                run_mode=True,
            )

        self.assertEqual(tuning.profile, "run")
        self.assertTrue(tuning.auto_gpu_utilization)
        self.assertTrue(tuning.auto_max_num_seqs)
        self.assertTrue(tuning.auto_max_model_len)
        self.assertTrue(tuning.auto_max_num_batched_tokens)
        self.assertTrue(tuning.auto_enforce_eager)
        self.assertEqual(tuning.gpu_memory_utilization, 0.80)
        self.assertEqual(tuning.max_num_seqs, 1)
        self.assertEqual(tuning.max_model_len, 8192)
        self.assertEqual(tuning.max_num_batched_tokens, 1024)
        self.assertTrue(tuning.enforce_eager)


if __name__ == "__main__":
    unittest.main()
