from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forgeai.cli.main import app
from forgeai.cli.runtime import DaemonClient

runner = CliRunner(mix_stderr=False)


class RunCommandTests(unittest.TestCase):
    @patch.object(DaemonClient, "stream")
    def test_run_streams_by_default(self, mock_stream: MagicMock) -> None:
        mock_stream.return_value = iter([
            {"model": "google/gemma-4-E2B-it", "response": "Hel", "done": False},
            {"model": "google/gemma-4-E2B-it", "response": "lo", "done": False},
            {"model": "google/gemma-4-E2B-it", "response": "", "done": True},
        ])

        with patch("forgeai.core.telemetry.track_event"):
            res = runner.invoke(app, ["run", "google/gemma-4-E2B-it", "Hello"])

        self.assertEqual(res.exit_code, 0)
        self.assertIn("Hello", res.stdout)
        mock_stream.assert_called_once_with(
            "POST",
            "/api/generate",
            json_data={
                "model": "google/gemma-4-E2B-it",
                "prompt": "Hello",
                "stream": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
