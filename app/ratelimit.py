import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """Per-key token bucket: `burst` capacity, refilled at `rpm` requests/minute."""

    def __init__(self, rpm: int, burst: int, clock: Callable[[], float] = time.monotonic):
        self.rate = rpm / 60.0  # tokens per second
        self.burst = burst
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire(self, key: str) -> tuple[bool, float]:
        """Try to take one token. Returns (allowed, retry_after_seconds)."""
        now = self.clock()
        with self._lock:
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(tokens=float(self.burst), updated=now)
                self._buckets[key] = b
            b.tokens = min(self.burst, b.tokens + (now - b.updated) * self.rate)
            b.updated = now
            if b.tokens >= 1:
                b.tokens -= 1
                return True, 0.0
            retry_after = (1 - b.tokens) / self.rate if self.rate > 0 else float("inf")
            return False, retry_after
