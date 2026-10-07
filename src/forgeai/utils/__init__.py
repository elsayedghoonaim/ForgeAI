"""Utility functions — GPU detection, VRAM estimation, and general helpers.

Imports are lazy (PEP 562) so importing a submodule does not load the whole package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forgeai.utils.helpers import (
        format_bytes as format_bytes,
    )
    from forgeai.utils.helpers import (
        format_duration as format_duration,
    )

_LAZY: dict[str, str] = {
    "format_bytes": "forgeai.utils.helpers",
    "format_duration": "forgeai.utils.helpers",
}

__all__ = [
    "format_bytes",
    "format_duration",
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
