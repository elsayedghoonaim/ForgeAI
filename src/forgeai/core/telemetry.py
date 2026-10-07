"""
Privacy-preserving, opt-in analytics.

Collects ZERO prompts or output data. Only tracks:
- Command invocations (which CLI command was run)
- Model names used
- Backend types (vllm)
- Error types (not content)

Disabled by default. Enable via FORGEAI_TELEMETRY_ENABLED=true.
"""

from __future__ import annotations

import atexit
import json
import os
import uuid
from collections import deque
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TelemetryCollector:
    """
    Lightweight, privacy-first telemetry with local-only storage.

    Events are stored as newline-delimited JSON in a local file.
    No data is transmitted anywhere unless explicitly configured.
    """

    MAX_BUFFERED_EVENTS = 1000

    def __init__(self, enabled: bool = False, storage_dir: str | None = None) -> None:
        self._enabled = enabled
        self._instance_id = str(uuid.uuid4())[:8]
        self._storage_dir = Path(
            storage_dir
            or os.path.join(os.path.expanduser("~"), ".forgeai", "telemetry")
        )
        self._events: deque[dict[str, Any]] = deque(maxlen=self.MAX_BUFFERED_EVENTS)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def track(
        self,
        event_name: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Record a telemetry event.

        Args:
            event_name: Name of the event (e.g., 'command.run', 'engine.init').
            properties: Additional metadata. Must NOT contain prompts or outputs.
        """
        if not self._enabled:
            return

        event = {
            "event": event_name,
            "instance_id": self._instance_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "properties": properties or {},
        }
        self._events.append(event)

    def flush(self) -> None:
        """Write buffered events to local storage."""
        if not self._enabled or not self._events:
            return

        self._storage_dir.mkdir(parents=True, exist_ok=True)
        log_file = self._storage_dir / f"events_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event) + "\n")

        self._events.clear()

    def disable(self) -> None:
        """Disable telemetry collection."""
        self._enabled = False
        self._events.clear()

    def shutdown(self) -> None:
        """Flush remaining events; safe to call repeatedly (registered with atexit)."""
        with suppress(Exception):
            self.flush()


# Global telemetry instance — disabled by default
_telemetry = TelemetryCollector(
    enabled=os.environ.get("FORGEAI_TELEMETRY_ENABLED", "false").lower() == "true"
)
atexit.register(_telemetry.shutdown)


def get_telemetry() -> TelemetryCollector:
    """Get the global telemetry collector."""
    return _telemetry


def track_event(event_name: str, properties: dict[str, Any] | None = None) -> None:
    """Convenience function to track an event."""
    _telemetry.track(event_name, properties)
