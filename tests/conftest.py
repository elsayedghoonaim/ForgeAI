"""Pytest compatibility hooks for the supported dependency range."""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner


# Click removed CliRunner(mix_stderr=...) while older ForgeAI tests still pass
# the argument. Typer delegates directly to Click, so accept and discard the
# obsolete test-only option until those call sites are modernized.
_original_cli_runner_init = CliRunner.__init__


def _cli_runner_init_compat(
    self: CliRunner,
    *args: Any,
    mix_stderr: bool | None = None,
    **kwargs: Any,
) -> None:
    del mix_stderr
    _original_cli_runner_init(self, *args, **kwargs)


CliRunner.__init__ = _cli_runner_init_compat  # type: ignore[method-assign]
