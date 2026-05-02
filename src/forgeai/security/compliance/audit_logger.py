"""Append-only audit logging with a tamper-evident hash chain."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class AuditLogger:
    """Append-only audit log with hash chaining for tamper detection."""

    def __init__(self, log_dir: str | None = None) -> None:
        self.log_dir = Path(
            log_dir or os.path.join(os.path.expanduser("~"), ".forgeai", "audit")
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._previous_hash = self._load_previous_hash()

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
                "timestamp": datetime.now(timezone.utc).isoformat(),
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
            self._previous_hash = entry_hash if isinstance(entry_hash, str) else "genesis"

            log_file = self.log_dir / f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
            with open(log_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")

            return entry

    def verify_chain(self, log_file: str) -> tuple[bool, int]:
        """Verify the integrity of an audit log file."""

        path = Path(log_file)
        if not path.exists():
            return True, 0

        prev_hash = "genesis"
        count = 0

        with open(path, encoding="utf-8") as handle:
            for line in handle:
                entry = self._parse_entry(line.strip())
                if entry.get("previous_hash") != prev_hash:
                    return False, count
                entry_hash = entry.get("hash")
                prev_hash = entry_hash if isinstance(entry_hash, str) else ""
                count += 1

        return True, count

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
