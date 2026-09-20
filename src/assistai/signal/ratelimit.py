"""Per-sender sliding-window rate limiting.

Inference is metered, so an allowed sender must not be able to run the bill up,
and an unknown sender must not be able to make the gateway emit unbounded
Signal traffic.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Allow ``limit`` events per ``window`` seconds, keyed by sender."""

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        max_keys: int = 256,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._limit = limit
        self._window = window_seconds
        self._max_keys = max_keys
        self._clock = clock or time.monotonic
        self._hits: dict[str, deque[float]] = {}

    def allow(self, key: str) -> bool:
        """Record an attempt. False once the key is over its budget."""
        hits = self._prepare(key)
        if len(hits) >= self._limit:
            return False
        hits.append(self._clock())
        return True

    def would_allow(self, key: str) -> bool:
        """True if ``allow`` would succeed. Does not spend the budget."""
        return len(self._prepare(key)) < self._limit

    def record(self, key: str) -> None:
        """Spend one slot after a successful send. Does not refuse."""
        self._prepare(key).append(self._clock())

    def _prepare(self, key: str) -> deque[float]:
        now = self._clock()
        self._evict(now)
        hits = self._hits.setdefault(key, deque())
        while hits and hits[0] <= now - self._window:
            hits.popleft()
        return hits

    def _evict(self, now: float) -> None:
        """Drop keys with no recent activity, then oldest, to bound memory."""
        if len(self._hits) < self._max_keys:
            return
        stale = [
            key for key, hits in self._hits.items() if not hits or hits[-1] <= now - self._window
        ]
        for key in stale:
            del self._hits[key]
        while len(self._hits) >= self._max_keys:
            oldest = min(self._hits, key=lambda key: self._hits[key][-1])
            del self._hits[oldest]
