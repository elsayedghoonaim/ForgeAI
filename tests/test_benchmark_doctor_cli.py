"""
Unit tests for benchmark CLI, doctor CLI, exact vLLM version enforcement, and diagnostic contracts.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from forgeai.cli.commands.doctor import (
    check_gpu_and_turboquant_diagnostic,
    check_platform_diagnostic,
    check_python_version,
    check_vllm_diagnostic,
)
from forgeai.cli.main import app
from forgeai.core.backends.vllm_backend import VLLMBackend
from forgeai.core.config import DevToolSettings
from forgeai.core.security import check_required_vllm_version


class BenchmarkCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_path = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_benchmark_default_plan_is_inert_and_emits_not_run(self) -> None:
        out_file = self.work_path / "plan_out.json"
        with patch("forgeai.core.telemetry.track_event"):
            res = self.runner.invoke(
                app,
                [
                    "benchmark",
                    "Qwen/Qwen3-0.6B",
                    "--mode",
                    "plan",
                    "--output",
                    str(out_file),
                    "--work-dir",
                    str(self.work_path),
                ],
            )

        self.assertEqual(res.exit_code, 0)
        self.assertTrue(out_file.exists())
        data = json.loads(out_file.read_text(encoding="utf-8"))
        self.assertEqual(data["model"], "Qwen/Qwen3-0.6B")
        self.assertEqual(data["status"], "not_run")
        self.assertFalse(data["hardware_validation_performed"])
        self.assertIn("command_plans", data)
        self.assertEqual(len(data["command_plans"]), 4)

    def test_benchmark_evaluate_handles_pass_and_non_pass_exit_codes(self) -> None:
        from forgeai.benchmarking.turboquant import (
            STATUS_NOT_RUN,
            STATUS_PASS,
            TurboQuantBenchmarkArtifact,
        )

        artifact = TurboQuantBenchmarkArtifact(model="Qwen/Qwen3-0.6B")
        artifact_file = self.work_path / "artifact.json"
        artifact_file.write_text(artifact.to_json(), encoding="utf-8")

        pass_result = TurboQuantBenchmarkArtifact.from_dict(artifact.to_dict())
        pass_result.status = STATUS_PASS
        with (
            patch("forgeai.core.telemetry.track_event"),
            patch("forgeai.benchmarking.turboquant.evaluate_artifact", return_value=pass_result),
        ):
            res_pass = self.runner.invoke(
                app,
                ["benchmark", "--mode", "evaluate", "--input", str(artifact_file)],
            )
        self.assertEqual(res_pass.exit_code, 0)

        non_pass_result = TurboQuantBenchmarkArtifact.from_dict(artifact.to_dict())
        non_pass_result.status = STATUS_NOT_RUN
        with (
            patch("forgeai.core.telemetry.track_event"),
            patch("forgeai.benchmarking.turboquant.evaluate_artifact", return_value=non_pass_result),
        ):
            res_fail = self.runner.invoke(
                app,
                ["benchmark", "--mode", "evaluate", "--input", str(artifact_file)],
            )
        self.assertEqual(res_fail.exit_code, 1)

    def test_benchmark_execute_without_acknowledgement_exits_before_runner(self) -> None:
        with patch("forgeai.benchmarking.runner.SequentialBenchmarkRunner") as mock_runner:
            with patch("forgeai.core.telemetry.track_event"):
                res = self.runner.invoke(app, ["benchmark", "--mode", "execute"])
            self.assertEqual(res.exit_code, 1)
            mock_runner.assert_not_called()

    def test_benchmark_unknown_mode_rejected(self) -> None:
        with patch("forgeai.core.telemetry.track_event"):
            res = self.runner.invoke(app, ["benchmark", "--mode", "unknown_mode"])
        self.assertEqual(res.exit_code, 1)

    def test_benchmark_gguf_rejected(self) -> None:
        with patch("forgeai.core.telemetry.track_event"):
            res = self.runner.invoke(app, ["benchmark", "model.gguf", "--mode", "plan"])
        self.assertEqual(res.exit_code, 1)

    def test_benchmark_base_port_validation_range_coverage(self) -> None:
        with patch("forgeai.core.telemetry.track_event"):
            res_valid = self.runner.invoke(
                app,
                ["benchmark", "Qwen/Qwen3-0.6B", "--mode", "plan", "--base-port", "65532"],
            )
            self.assertEqual(res_valid.exit_code, 0)

            res_high1 = self.runner.invoke(
                app,
                ["benchmark", "Qwen/Qwen3-0.6B", "--mode", "plan", "--base-port", "65533"],
            )
            self.assertEqual(res_high1.exit_code, 1)

            res_high2 = self.runner.invoke(
                app,
                ["benchmark", "Qwen/Qwen3-0.6B", "--mode", "plan", "--base-port", "65535"],
            )
            self.assertEqual(res_high2.exit_code, 1)

            res_zero = self.runner.invoke(
                app,
                ["benchmark", "Qwen/Qwen3-0.6B", "--mode", "plan", "--base-port", "0"],
            )
            self.assertEqual(res_zero.exit_code, 1)

            res_neg = self.runner.invoke(
                app,
                ["benchmark", "Qwen/Qwen3-0.6B", "--mode", "plan", "--base-port", "-10"],
            )
            self.assertEqual(res_neg.exit_code, 1)


class ExactVllmVersionCheckerTests(unittest.TestCase):
    def test_accepts_exact_0_29_0_and_local_build_metadata(self) -> None:
        for ver in ["0.29.0", "0.29.0+cu129", "0.29.0+rocm6.0"]:
            mock_vllm = ModuleType("vllm")
            mock_vllm.__version__ = ver
            with self.subTest(version=ver):
                self.assertTrue(
                    check_required_vllm_version(vllm_module=mock_vllm, announce_success=False)
                )

    def test_rejects_missing_and_unknown_vllm(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            check_required_vllm_version(vllm_module=False, announce_success=False)
        self.assertIn("vLLM is not installed", str(ctx.exception))

        mock_vllm = ModuleType("vllm")
        mock_vllm.__version__ = None
        with self.assertRaises(RuntimeError) as ctx:
            check_required_vllm_version(vllm_module=mock_vllm, announce_success=False)
        self.assertIn("Cannot determine vLLM version", str(ctx.exception))

    def test_rejects_older_versions(self) -> None:
        for ver in ["0.14.0", "0.28.0"]:
            mock_vllm = ModuleType("vllm")
            mock_vllm.__version__ = ver
            with self.subTest(version=ver):
                with self.assertRaises(RuntimeError) as ctx:
                    check_required_vllm_version(vllm_module=mock_vllm, announce_success=False)
                self.assertIn("ForgeAI requires exact vLLM version 0.29.0", str(ctx.exception))

    def test_rejects_newer_versions(self) -> None:
        for ver in ["0.29.1", "0.30.0"]:
            mock_vllm = ModuleType("vllm")
            mock_vllm.__version__ = ver
            with self.subTest(version=ver):
                with self.assertRaises(RuntimeError) as ctx:
                    check_required_vllm_version(vllm_module=mock_vllm, announce_success=False)
                self.assertIn("ForgeAI requires exact vLLM version 0.29.0", str(ctx.exception))

    def test_rejects_prerelease_and_dev_versions(self) -> None:
        for ver in ["0.29.0rc1", "0.29.0.dev0"]:
            mock_vllm = ModuleType("vllm")
            mock_vllm.__version__ = ver
            with self.subTest(version=ver):
                with self.assertRaises(RuntimeError) as ctx:
                    check_required_vllm_version(vllm_module=mock_vllm, announce_success=False)
                self.assertIn("ForgeAI requires exact vLLM version 0.29.0", str(ctx.exception))

    def test_rejects_malformed_versions(self) -> None:
        for ver in ["not-a-valid-version", "0.29.0.invalid!"]:
            mock_vllm = ModuleType("vllm")
            mock_vllm.__version__ = ver
            with self.subTest(version=ver):
                with self.assertRaises(RuntimeError) as ctx:
                    check_required_vllm_version(vllm_module=mock_vllm, announce_success=False)
                self.assertIn("Invalid vLLM version string format", str(ctx.exception))


class BackendVersionEnforcementTests(unittest.TestCase):
    @patch("forgeai.core.backends.vllm_backend.validate_turboquant_hardware")
    @patch.object(VLLMBackend, "_preflight_vllm_memory")
    @patch.object(VLLMBackend, "_init_vllm")
    @patch("forgeai.core.backends.vllm_backend.check_required_vllm_version")
    def test_vllm_backend_calls_strict_checker(
        self,
        mock_strict_checker: MagicMock,
        mock_init_vllm: MagicMock,
        mock_preflight: MagicMock,
        mock_val_hw: MagicMock,
    ) -> None:
        settings = DevToolSettings(
            model_name="Qwen/Qwen3-0.6B",
            enforce_version_check=True,
        )
        backend = VLLMBackend(settings)
        backend.initialize()
        mock_strict_checker.assert_called_once()


class DoctorDiagnosticContractTests(unittest.TestCase):
    def test_doctor_python_version_check(self) -> None:
        ok, detail = check_python_version((3, 12, 1))
        self.assertTrue(ok)
        self.assertIn("3.12.1", detail)

        ok, detail = check_python_version((3, 11, 9))
        self.assertFalse(ok)
        self.assertIn("Upgrade to Python 3.12", detail)

        ok, detail = check_python_version((3, 13, 0))
        self.assertFalse(ok)

    def test_doctor_vllm_exact_check(self) -> None:
        for ver in ["0.29.0", "0.29.0+cu129", "0.29.0+rocm6.0"]:
            mock_vllm = ModuleType("vllm")
            mock_vllm.__version__ = ver
            with self.subTest(version=ver):
                ok, detail = check_vllm_diagnostic(mock_vllm)
                self.assertTrue(ok)
                self.assertEqual(detail, f"Installed: {ver}")

        mock_vllm = ModuleType("vllm")
        mock_vllm.__version__ = "0.14.0"
        ok, detail = check_vllm_diagnostic(mock_vllm)
        self.assertFalse(ok)
        self.assertNotIn("optional", detail.lower())

        mock_vllm.__version__ = "not-a-valid-version"
        ok, detail = check_vllm_diagnostic(mock_vllm)
        self.assertFalse(ok)
        self.assertIn("Installed: not-a-valid-version → pip install 'vllm==0.29.0'", detail)

        mock_vllm.__version__ = None
        ok, detail = check_vllm_diagnostic(mock_vllm)
        self.assertFalse(ok)
        self.assertIn("Installed: unknown → pip install 'vllm==0.29.0'", detail)

        ok, detail = check_vllm_diagnostic(vllm_module=False)
        self.assertFalse(ok)
        self.assertNotIn("optional", detail.lower())

    def test_doctor_platform_diagnostic(self) -> None:
        ok, detail = check_platform_diagnostic(system_name="Linux", is_wsl=False)
        self.assertTrue(ok)
        self.assertIn("Linux (native supported)", detail)

        ok, detail = check_platform_diagnostic(system_name="Linux", is_wsl=True)
        self.assertTrue(ok)
        self.assertIn("Linux (WSL2 supported)", detail)

        ok, detail = check_platform_diagnostic(system_name="Windows")
        self.assertFalse(ok)
        self.assertIn("Windows unsupported", detail)

        ok, detail = check_platform_diagnostic(system_name="Darwin")
        self.assertFalse(ok)
        self.assertIn("Darwin unsupported", detail)

    def test_doctor_cuda_rocm_cpu_diagnostics(self) -> None:
        mock_gpu = MagicMock()
        mock_gpu.name = "NVIDIA A100"
        mock_gpu.compute_capability = (8, 0)
        mock_topo = MagicMock()
        mock_topo.gpu_count = 1
        mock_topo.gpus = [mock_gpu]

        checks = check_gpu_and_turboquant_diagnostic(gpu_topology=mock_topo, is_rocm=False)
        self.assertEqual(len(checks), 2)
        self.assertTrue(checks[0][1])
        self.assertTrue(checks[1][1])
        self.assertIn("CC 8.0 >= 7.5", checks[1][2])

        mock_gpu_old = MagicMock()
        mock_gpu_old.name = "NVIDIA V100"
        mock_gpu_old.compute_capability = (7, 0)
        mock_topo_old = MagicMock()
        mock_topo_old.gpu_count = 1
        mock_topo_old.gpus = [mock_gpu_old]

        checks = check_gpu_and_turboquant_diagnostic(gpu_topology=mock_topo_old, is_rocm=False)
        self.assertTrue(checks[0][1])
        self.assertFalse(checks[1][1])
        self.assertIn("CC 7.0 < 7.5", checks[1][2])

        mock_gpu_unk = MagicMock()
        mock_gpu_unk.name = "Unknown GPU"
        mock_gpu_unk.compute_capability = (0, 0)
        mock_topo_unk = MagicMock()
        mock_topo_unk.gpu_count = 1
        mock_topo_unk.gpus = [mock_gpu_unk]

        checks = check_gpu_and_turboquant_diagnostic(gpu_topology=mock_topo_unk, is_rocm=False)
        self.assertTrue(checks[0][1])
        self.assertFalse(checks[1][1])
        self.assertIn("missing/unknown", checks[1][2].lower())

        mock_topo_rocm = MagicMock()
        mock_topo_rocm.gpu_count = 1
        checks = check_gpu_and_turboquant_diagnostic(gpu_topology=mock_topo_rocm, is_rocm=True)
        self.assertTrue(checks[0][1])
        self.assertFalse(checks[1][1])
        self.assertIn("ROCm", checks[1][2])

        mock_topo_cpu = MagicMock()
        mock_topo_cpu.gpu_count = 0
        mock_topo_cpu.gpus = []
        checks = check_gpu_and_turboquant_diagnostic(gpu_topology=mock_topo_cpu, is_rocm=False)
        self.assertFalse(checks[0][1])
        self.assertFalse(checks[1][1])


if __name__ == "__main__":
    unittest.main()
