from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the src folder is on Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.core.security import (
    check_vllm_version,
    sanitize_path,
    validate_parallelism,
)
from forgeai.models.loader import (
    _is_valid_repo_id,
    delete_cached_model,
    download_model,
)
from forgeai.models.safety_scanner import scan_model_weights


class SecurityPathSanitizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sanitize_path_valid(self) -> None:
        # Path inside base is allowed
        safe_path = self.base_path / "model.safetensors"
        resolved = sanitize_path(str(safe_path), allowed_base=str(self.base_path))
        self.assertEqual(resolved, safe_path.resolve())

    def test_sanitize_path_relative_valid(self) -> None:
        # Path with ".." resolving within base is allowed
        relative_safe = self.base_path / "subdir" / ".." / "model.safetensors"
        resolved = sanitize_path(str(relative_safe), allowed_base=str(self.base_path))
        self.assertEqual(resolved, (self.base_path / "model.safetensors").resolve())

    def test_sanitize_path_traversal_escape(self) -> None:
        # Path escaping base is rejected
        outside_path = self.base_path / ".." / "outside.txt"
        with self.assertRaises(ValueError) as ctx:
            sanitize_path(str(outside_path), allowed_base=str(self.base_path))
        self.assertIn("Path escapes allowed directory", str(ctx.exception))

    def test_sanitize_path_prefix_bypass_rejected(self) -> None:
        # Prefix bypass: /tmp/allowed_base-sibling starts with /tmp/allowed_base but is outside
        sibling_base = self.base_path.parent / (self.base_path.name + "-sibling")
        sibling_base.mkdir(exist_ok=True)
        try:
            outside_path = sibling_base / "model.safetensors"
            with self.assertRaises(ValueError) as ctx:
                sanitize_path(str(outside_path), allowed_base=str(self.base_path))
            self.assertIn("Path escapes allowed directory", str(ctx.exception))
        except ValueError as e:
            self.assertIn("Path escapes allowed directory", str(e))
        finally:
            if sibling_base.exists():
                import shutil
                shutil.rmtree(sibling_base)

    def test_sanitize_path_no_base_traversal_rejected(self) -> None:
        # Traversal in input is rejected if allowed_base is None
        with self.assertRaises(ValueError) as ctx:
            sanitize_path("foo/../bar", allowed_base=None)
        self.assertIn("Path traversal detected", str(ctx.exception))


class SecurityVllmVersionTests(unittest.TestCase):
    def test_version_below_minimum(self) -> None:
        import sys
        from types import ModuleType
        mock_vllm = ModuleType("vllm")
        mock_vllm.__version__ = "0.13.9"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with self.assertRaises(RuntimeError) as ctx:
                check_vllm_version(min_version="0.14.0", announce_success=False)
            self.assertIn("SECURITY: vLLM 0.13.9 is vulnerable to CVE-2026-22807", str(ctx.exception))

    def test_version_equal_to_minimum(self) -> None:
        import sys
        from types import ModuleType
        mock_vllm = ModuleType("vllm")
        mock_vllm.__version__ = "0.14.0"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            self.assertTrue(check_vllm_version(min_version="0.14.0", announce_success=False))

    def test_version_prerelease_below_minimum(self) -> None:
        import sys
        from types import ModuleType
        mock_vllm = ModuleType("vllm")
        # 0.14.0rc1 is below 0.14.0
        mock_vllm.__version__ = "0.14.0rc1"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with self.assertRaises(RuntimeError) as ctx:
                check_vllm_version(min_version="0.14.0", announce_success=False)
            self.assertIn("SECURITY: vLLM 0.14.0rc1 is vulnerable to CVE-2026-22807", str(ctx.exception))

    def test_version_prerelease_above_minimum(self) -> None:
        import sys
        from types import ModuleType
        mock_vllm = ModuleType("vllm")
        # 0.14.1rc1 is above 0.14.0
        mock_vllm.__version__ = "0.14.1rc1"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            self.assertTrue(check_vllm_version(min_version="0.14.0", announce_success=False))

    def test_version_above_minimum(self) -> None:
        import sys
        from types import ModuleType
        mock_vllm = ModuleType("vllm")
        mock_vllm.__version__ = "0.15.0"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            self.assertTrue(check_vllm_version(min_version="0.14.0", announce_success=False))

    def test_version_malformed_string(self) -> None:
        import sys
        from types import ModuleType
        mock_vllm = ModuleType("vllm")
        mock_vllm.__version__ = "not-a-valid-version-string"

        with patch.dict(sys.modules, {"vllm": mock_vllm}):
            with self.assertRaises(RuntimeError) as ctx:
                check_vllm_version(min_version="0.14.0", announce_success=False)
            self.assertIn("Invalid vLLM version string format", str(ctx.exception))


class SecurityRepoIdTests(unittest.TestCase):
    def test_valid_repo_ids(self) -> None:
        valid_cases = [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "gpt2",
            "google/gemma-4-E2B-it",
            "some-user/model_name.v1",
            "a/b",
            "a-b/c_d.e",
        ]
        for case in valid_cases:
            with self.subTest(case=case):
                self.assertTrue(_is_valid_repo_id(case), f"Should be valid: {case}")

    def test_invalid_repo_ids(self) -> None:
        invalid_cases = [
            "meta-llama/Meta-Llama/8B",  # Too many slashes
            "../../etc/passwd",          # Path traversal
            "gpt2; rm -rf /",            # Command injection
            "user/model$",               # Unsafe characters
            "-user/model",               # Starts with dash
            "user/model-",               # Ends with dash
            ".user/model",               # Starts with dot
            "user/model.",               # Ends with dot
            "user//model",               # Consecutive slashes
            "a" * 201,                   # Too long
            "",                          # Empty
        ]
        for case in invalid_cases:
            with self.subTest(case=case):
                self.assertFalse(_is_valid_repo_id(case), f"Should be invalid: {case}")

    def test_download_model_validates_repo_id(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            download_model("invalid; injection")
        self.assertIn("Invalid HuggingFace repository ID format", str(ctx.exception))

    def test_delete_cached_model_validates_repo_id(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            delete_cached_model("invalid; injection")
        self.assertIn("Invalid HuggingFace repository ID format", str(ctx.exception))

    @patch("huggingface_hub.snapshot_download")
    def test_download_model_fail_closed(self, mock_snapshot_download) -> None:
        # Create a temp dir to act as the download location
        temp_download_dir = tempfile.TemporaryDirectory()
        download_path = Path(temp_download_dir.name).resolve()

        # Write a malicious weight file inside the download location
        class Malicious:
            def __reduce__(self):
                import os
                return (os.system, ("echo unsafe",))

        unsafe_stream = pickle.dumps(Malicious(), protocol=4)
        file_path = download_path / "model.bin"
        with zipfile.ZipFile(file_path, "w") as z:
            z.writestr("archive/data.pkl", unsafe_stream)

        mock_snapshot_download.return_value = str(download_path)
        mock_hf = MagicMock()
        mock_hf.snapshot_download.return_value = str(download_path)

        try:
            with patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
                # Call download_model, it should download, scan, fail and raise ValueError
                with self.assertRaises(ValueError) as ctx:
                    download_model(
                        repo_id="gpt2",
                        cache_dir=str(download_path.parent),
                        enable_safety_scan=True
                    )
                self.assertIn("SECURITY BLOCK: Model safety scan failed", str(ctx.exception))

                # Check that the download directory was deleted/cleaned up
                self.assertFalse(download_path.exists())
        finally:
            if download_path.exists():
                import shutil
                shutil.rmtree(download_path)
            temp_download_dir.cleanup()


class SecurityParallelismTests(unittest.TestCase):
    def test_validate_parallelism_valid(self) -> None:
        self.assertEqual(validate_parallelism(2, 4), 2)
        self.assertEqual(validate_parallelism(1, 1), 1)

    def test_validate_parallelism_less_than_one(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_parallelism(0, 4)
        self.assertIn("tensor_parallel_size must be >= 1", str(ctx.exception))

    def test_validate_parallelism_exceeds_available(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            validate_parallelism(8, 4)
        self.assertIn("exceeds available GPUs", str(ctx.exception))

    def test_validate_parallelism_non_power_of_two_warning(self) -> None:
        with patch("rich.console.Console.print") as mock_print:
            validate_parallelism(3, 4)
            mock_print.assert_called_once()
            args, kwargs = mock_print.call_args
            self.assertIn("is not a power of 2", args[0])


class SecuritySafetyScannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_scan_pytorch_format_warning_and_safe_pickle(self) -> None:
        # Test safe zip pickle
        safe_stream = pickle.dumps({"weights": [1.0, 2.0]})

        file_path = self.dir_path / "model.bin"
        with zipfile.ZipFile(file_path, "w") as z:
            z.writestr("archive/data.pkl", safe_stream)

        result = scan_model_weights(str(file_path))

        self.assertTrue(result["safe"])
        self.assertTrue(any("Insecure PyTorch format" in w for w in result["warnings"]))
        self.assertFalse(any("Unsafe pickle" in w for w in result["warnings"]))

    def test_scan_pytorch_malicious_pickle(self) -> None:
        # Test unsafe zip pickle using custom object reducing to posix.system
        class Malicious:
            def __reduce__(self):
                import os
                return (os.system, ("echo unsafe",))

        unsafe_stream = pickle.dumps(Malicious(), protocol=0)

        file_path = self.dir_path / "model.bin"
        with zipfile.ZipFile(file_path, "w") as z:
            z.writestr("archive/data.pkl", unsafe_stream)

        result = scan_model_weights(str(file_path))

        self.assertFalse(result["safe"])
        self.assertTrue(any("Insecure PyTorch format" in w for w in result["warnings"]))
        self.assertTrue(any("Unsafe pickle global" in w for w in result["warnings"]))
        self.assertTrue(any("os.system" in w for w in result["warnings"]))

    def test_scan_pytorch_malicious_pickle_protocol_4_stack_global(self) -> None:
        # Test unsafe pickle using protocol 4 (which uses STACK_GLOBAL)
        class Malicious:
            def __reduce__(self):
                import os
                return (os.system, ("echo unsafe",))

        unsafe_stream = pickle.dumps(Malicious(), protocol=4)

        file_path = self.dir_path / "model_proto4.bin"
        with zipfile.ZipFile(file_path, "w") as z:
            z.writestr("archive/data.pkl", unsafe_stream)

        result = scan_model_weights(str(file_path))

        self.assertFalse(result["safe"])
        self.assertTrue(any("Unsafe pickle global" in w for w in result["warnings"]))
        self.assertTrue(any("os.system" in w for w in result["warnings"]))


class SecurityAbsolutePathsTests(unittest.TestCase):
    def test_sanitize_path_absolute_no_base_rejected(self) -> None:
        # Absolute paths are rejected when allowed_base is None
        with self.assertRaises(ValueError) as ctx:
            sanitize_path("/etc/passwd", allowed_base=None)
        self.assertIn("Absolute paths not permitted when allowed_base is None", str(ctx.exception))


class AuditLoggerVerifyChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name).resolve()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_audit_logger_chain_tamper_detection(self) -> None:
        from forgeai.security.compliance.audit_logger import AuditLogger

        logger = AuditLogger(log_dir=str(self.log_dir))
        logger.log("access", "admin", "POST", "/v1/chat/completions", outcome="success")
        logger.log("access", "admin", "GET", "/metrics", outcome="success")

        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        self.assertEqual(len(log_files), 1)
        log_file = log_files[0]

        # 1. Verification of untampered log file passes
        ok, count = logger.verify_chain(str(log_file))
        self.assertTrue(ok)
        self.assertEqual(count, 2)

        # 2. Tampering a record payload (e.g. changing outcome to Success) causes verification failure
        with open(log_file, encoding="utf-8") as f:
            lines = f.readlines()

        import json
        entry = json.loads(lines[0].strip())
        entry["outcome"] = "tampered_success"  # Modify value without changing stored hash

        lines[0] = json.dumps(entry) + "\n"
        with open(log_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Chain verification must now fail
        ok, count = logger.verify_chain(str(log_file))
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()

