"""Simple in-memory rate limiting primitives."""

from __future__ import annotations

import math
import time
from collections import deque
from threading import Lock


class MemoryRateLimiter:
    """Sliding-window in-memory rate limiter."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = {}
        self._lock = Lock()
        self._last_sweep = time.monotonic()

    def check(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)`` for the given key."""

        current_time = now if now is not None else time.monotonic()
        with self._lock:
            self._sweep(current_time)
            events = self._events.setdefault(key, deque())
            while events and current_time - events[0] >= self.window_seconds:
                events.popleft()

            if len(events) >= self.limit:
                retry_after = max(1, math.ceil(self.window_seconds - (current_time - events[0])))
                return False, retry_after

            events.append(current_time)
            return True, 0

    def _sweep(self, current_time: float) -> None:
        """Drop keys whose events have all expired (called with the lock held)."""

        if current_time - self._last_sweep < self.window_seconds:
            return
        self._last_sweep = current_time
        for key in [
            k
            for k, ev in self._events.items()
            if not ev or current_time - ev[-1] >= self.window_seconds
        ]:
            del self._events[key]
