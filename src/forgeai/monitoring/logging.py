"""Structured logging configuration with JSON output and Rich console handler."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "method",
            "path",
            "status",
            "duration_ms",
            "model",
            "error",
            "request_id",
            "actor",
            "permission",
            "client_ip",
        ):
            value = getattr(record, key, None)
            if value is not None:
                log_data[key] = value

        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = str(record.exc_info[1])

        return json.dumps(log_data)


class StructuredLogger:
    """Logger wrapper that supports structured key-value logging."""

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)

    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        self._logger.log(level, message, extra=kwargs, stacklevel=3)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, message, **kwargs)


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure the logging system."""

    root = logging.getLogger("forgeai")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    handler: logging.Handler

    if json_output:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
    else:
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
        except ImportError:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )

    root.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module."""

    return StructuredLogger(f"forgeai.{name}")
