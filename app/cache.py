import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

from app.schemas import ChatRequest, ChatResponse


def cache_key(request: ChatRequest) -> str:
    blob = json.dumps(request.model_dump(), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def is_cacheable(request: ChatRequest) -> bool:
    """Only deterministic requests are safe to serve from cache."""
    return request.temperature in (None, 0, 0.0)


class ResponseCache:
    """Exact-match LRU cache with TTL. (Semantic cache is on the roadmap.)"""

    def __init__(self, ttl: int, max_entries: int, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self._data: OrderedDict[str, tuple[float, ChatResponse]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[ChatResponse]:
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

    def set(self, key: str, value: ChatResponse) -> None:
        with self._lock:
            self._data[key] = (self.clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
