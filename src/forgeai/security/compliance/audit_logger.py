"""Append-only audit logging with a tamper-evident hash chain."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_STOP = object()


class AuditLogger:
    """Append-only audit log with hash chaining for tamper detection."""

    def __init__(self, log_dir: str | None = None, max_queue_size: int = 10000) -> None:
        self.log_dir = Path(
            log_dir or os.path.join(os.path.expanduser("~"), ".forgeai", "audit")
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._previous_hash = self._load_previous_hash()
        self.dropped_events = 0
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._writer = threading.Thread(
            target=self._writer_loop, name="forgeai-audit-writer", daemon=True
        )
        self._writer.start()

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    return
                path, line = item
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line)
            except OSError:
                logger.exception("audit_write_failed")
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Block until all queued events are written to disk."""

        self._queue.join()

    def close(self) -> None:
        """Flush pending events and stop the writer thread."""

        if self._writer.is_alive():
            self._queue.put(_STOP)
            self._writer.join(timeout=10)

    def _load_previous_hash(self) -> str:
        latest_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        if not latest_files:
            return "genesis"

        latest_file = latest_files[-1]
        try:
            last_line = ""
            with open(latest_file, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last_line = line.strip()
            if not last_line:
                return "genesis"
            entry = self._parse_entry(last_line)
            entry_hash = entry.get("hash")
            return entry_hash if isinstance(entry_hash, str) else "genesis"
        except (OSError, ValueError, json.JSONDecodeError):
            return "genesis"

    def log(
        self,
        event_type: str,
        actor: str,
        action: str,
        resource: str = "",
        details: dict[str, Any] | None = None,
        outcome: str = "success",
    ) -> dict[str, Any]:
        """Log a security-relevant event."""

        with self._lock:
            entry: dict[str, Any] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event_type": event_type,
                "actor": actor,
                "action": action,
                "resource": resource,
                "outcome": outcome,
                "details": details or {},
                "previous_hash": self._previous_hash,
            }

            entry_str = json.dumps(entry, sort_keys=True)
            entry["hash"] = hashlib.sha256(entry_str.encode()).hexdigest()
            entry_hash = entry["hash"]

            log_file = self.log_dir / f"audit_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
            try:
                self._queue.put_nowait((log_file, json.dumps(entry) + "\n"))
            except queue.Full:
                # Do not advance the chain for an event that was never written.
                self.dropped_events += 1
                logger.error("audit_queue_full_event_dropped")
                return entry
            self._previous_hash = entry_hash if isinstance(entry_hash, str) else "genesis"
            return entry

    def verify_chain(self, log_file: str) -> tuple[bool, int]:
        """Verify the integrity of an audit log file and its cryptographic hash chain."""

        path = Path(log_file)
        if not path.exists():
            return True, 0

        prev_hash = self._hash_before(path)
        count = 0

        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = self._parse_entry(line.strip())

                # 1. Verify continuity of the cryptographic chain
                if entry.get("previous_hash") != prev_hash:
                    return False, count

                # 2. Extract and verify stored signature against content digest
                stored_hash = entry.pop("hash", None)
                if not stored_hash:
                    return False, count

                recalc_str = json.dumps(entry, sort_keys=True)
                recalculated_hash = hashlib.sha256(recalc_str.encode()).hexdigest()

                if recalculated_hash != stored_hash:
                    return False, count

                prev_hash = stored_hash
                count += 1

        return True, count


    def _hash_before(self, path: Path) -> str:
        """Return the chain hash that must precede the first entry of ``path``.

        The writer carries the previous hash across daily files, so the verifier
        starts from the last hash of the preceding file in the same directory.
        """

        siblings = sorted(path.parent.glob("audit_*.jsonl"))
        earlier = [p for p in siblings if p.name < path.name]
        for candidate in reversed(earlier):
            last = ""
            with open(candidate, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        last = line.strip()
            if last:
                value = self._parse_entry(last).get("hash")
                if isinstance(value, str):
                    return value
        return "genesis"

    def query(
        self,
        event_type: str | None = None,
        actor: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query audit logs with filters."""

        results: list[dict[str, Any]] = []
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file, encoding="utf-8") as handle:
                for line in handle:
                    entry = self._parse_entry(line.strip())
                    if event_type and entry.get("event_type") != event_type:
                        continue
                    if actor and entry.get("actor") != actor:
                        continue
                    if start_date and entry.get("timestamp", "") < start_date:
                        continue
                    if end_date and entry.get("timestamp", "") > end_date:
                        continue
                    results.append(entry)

        return results

    @staticmethod
    def _parse_entry(line: str) -> dict[str, Any]:
        parsed = json.loads(line)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Audit log entry must be a JSON object.")
