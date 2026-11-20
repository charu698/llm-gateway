"""Per-key token-bucket rate limiting: in-process or Redis (shared across instances)."""
import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

log = logging.getLogger("gateway.ratelimit")


class Limiter(Protocol):
    kind: str

    async def acquire(self, key: str) -> tuple[bool, float]:
        """Try to take one token. Returns (allowed, retry_after_seconds)."""


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    """`burst` capacity, refilled at `rpm` requests/minute. Local to one process."""

    kind = "memory"

    def __init__(self, rpm: int, burst: int, clock: Callable[[], float] = time.monotonic):
        self.rate = rpm / 60.0  # tokens per second
        self.burst = burst
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def acquire_sync(self, key: str) -> tuple[bool, float]:
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

    async def acquire(self, key: str) -> tuple[bool, float]:
        return self.acquire_sync(key)


# Atomic token bucket. Uses Redis's own clock (TIME) so instances with skewed
# clocks still agree, and a single script call so concurrent requests can't race.
_LUA_TOKEN_BUCKET = """
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local data = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = burst
  ts = now
end
tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
local retry = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
elseif rate > 0 then
  retry = (1 - tokens) / rate
end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
if rate > 0 then
  redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) * 2 + 1)
end
return {allowed, tostring(retry)}
"""


class RedisTokenBucketLimiter:
    """Distributed token bucket. Fails open: if Redis is down, requests are allowed."""

    kind = "redis"

    def __init__(self, client, rpm: int, burst: int, prefix: str = "gw:rl:"):
        self.client = client
        self.rate = rpm / 60.0
        self.burst = burst
        self.prefix = prefix
        self._script = client.register_script(_LUA_TOKEN_BUCKET)

    async def acquire(self, key: str) -> tuple[bool, float]:
        try:
            allowed, retry = await self._script(
                keys=[self.prefix + key], args=[self.rate, self.burst]
            )
        except Exception as exc:
            log.warning("redis rate limiter unavailable, failing open: %s", exc)
            return True, 0.0
        return bool(int(allowed)), float(retry)
