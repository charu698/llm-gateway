"""Response caching (in-process or Redis) and in-flight request coalescing."""
import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Awaitable, Callable, Optional, Protocol, TypeVar

from app.schemas import ChatRequest, ChatResponse

log = logging.getLogger("gateway.cache")
T = TypeVar("T")


def cache_key(request: ChatRequest) -> str:
    # `stream` is excluded so streamed and non-streamed calls share cache entries.
    blob = json.dumps(request.model_dump(exclude={"stream"}), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def is_cacheable(request: ChatRequest) -> bool:
    """Only deterministic requests are safe to serve from cache."""
    return request.temperature in (None, 0, 0.0)


class CacheBackend(Protocol):
    kind: str

    async def get(self, key: str) -> Optional[ChatResponse]: ...
    async def set(self, key: str, value: ChatResponse) -> None: ...


class MemoryCache:
    """Exact-match LRU cache with TTL, local to one process."""

    kind = "memory"

    def __init__(self, ttl: int, max_entries: int, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self._data: OrderedDict[str, tuple[float, ChatResponse]] = OrderedDict()
        self._lock = threading.Lock()

    async def get(self, key: str) -> Optional[ChatResponse]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            stored_at, value = item
            if self.clock() - stored_at > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: ChatResponse) -> None:
        with self._lock:
            self._data[key] = (self.clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)


class RedisCache:
    """Cache shared by every gateway instance. Fails open: a Redis outage is a miss."""

    kind = "redis"

    def __init__(self, client, ttl: int, prefix: str = "gw:cache:"):
        self.client = client
        self.ttl = ttl
        self.prefix = prefix

    async def get(self, key: str) -> Optional[ChatResponse]:
        try:
            raw = await self.client.get(self.prefix + key)
        except Exception as exc:  # redis.RedisError, connection refused, timeouts
            log.warning("redis cache get failed, treating as miss: %s", exc)
            return None
        return ChatResponse.model_validate_json(raw) if raw else None

    async def set(self, key: str, value: ChatResponse) -> None:
        try:
            await self.client.set(self.prefix + key, value.model_dump_json(), ex=self.ttl)
        except Exception as exc:
            log.warning("redis cache set failed, skipping: %s", exc)


class _LeaderCancelled(Exception):
    pass


class SingleFlight:
    """Collapse concurrent identical requests into one upstream call.

    The first caller (the leader) does the work; callers arriving while it is
    in flight wait for the same result instead of each hitting the provider.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future] = {}

    async def run(self, key: str, fn: Callable[[], Awaitable[T]]) -> tuple[T, bool]:
        """Returns (result, coalesced). `coalesced` is True for followers."""
        while True:
            fut = self._inflight.get(key)
            if fut is None:
                break
            try:
                return await asyncio.shield(fut), True
            except _LeaderCancelled:
                continue  # leader's client went away; become the new leader

        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await fn()
        except asyncio.CancelledError:
            fut.set_exception(_LeaderCancelled())
            fut.exception()  # mark retrieved so asyncio doesn't warn
            raise
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()
            raise
        else:
            fut.set_result(result)
            return result, False
        finally:
            self._inflight.pop(key, None)
