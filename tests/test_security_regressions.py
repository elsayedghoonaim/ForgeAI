from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forgeai.cli.commands.serve import _is_loopback_host
from forgeai.models.loader import CacheManager


class SecurityRegressionTests(unittest.TestCase):
    def test_remote_bind_detection(self) -> None:
        self.assertTrue(_is_loopback_host("127.0.0.1"))
        self.assertTrue(_is_loopback_host("::1"))
        self.assertTrue(_is_loopback_host("localhost"))
        self.assertFalse(_is_loopback_host("0.0.0.0"))
        self.assertFalse(_is_loopback_host("192.168.1.20"))

    def test_secure_snapshot_rejects_pickle_weights_and_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot = root / "hf" / "hub" / "models--org--model" / "snapshots" / "abc"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.bin").write_bytes(b"not-a-real-model")

            manager = CacheManager(
                forgeai_home=root / "forgeai",
                hf_home=root / "hf",
            )
            with patch.object(manager, "get_snapshot_path", return_value=snapshot):
                with self.assertRaisesRegex(ValueError, "Pickle-backed model weights"):
                    manager.download_snapshot_secure("org/model", enable_safety_scan=False)

            self.assertFalse(snapshot.exists())

    def test_secure_snapshot_requires_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            snapshot = root / "hf" / "hub" / "models--org--model" / "snapshots" / "abc"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            manager = CacheManager(
                forgeai_home=root / "forgeai",
                hf_home=root / "hf",
            )
            with patch.object(manager, "get_snapshot_path", return_value=snapshot):
                with self.assertRaisesRegex(ValueError, "no .safetensors"):
                    manager.download_snapshot_secure("org/model", enable_safety_scan=False)

            self.assertFalse(snapshot.exists())


if __name__ == "__main__":
    unittest.main()
