"""Inference backends package.

Imports are lazy (PEP 562) so importing a submodule does not load the whole package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forgeai.core.backends.base import (
        BaseBackend as BaseBackend,
    )
    from forgeai.core.backends.factory import (
        create_backend as create_backend,
    )

_LAZY: dict[str, str] = {
    "BaseBackend": "forgeai.core.backends.base",
    "create_backend": "forgeai.core.backends.factory",
}

__all__ = [
    "BaseBackend",
    "create_backend",
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
