"""Request log and usage stats on SQLAlchemy: SQLite for dev, Postgres for production.

Writes are queued and flushed in batches by a background thread, so logging
never adds a database round trip to the request path.
"""
import logging
import math
import queue
import threading
import time
from typing import Optional

from sqlalchemy import (
    Boolean, Column, Float, Index, Integer, MetaData, String, Table, case, create_engine,
    event, func, insert, select,
)

log = logging.getLogger("gateway.db")

# USD per 1M tokens as (input, output). Models not listed are costed at 0.
# Fill these in from your provider's pricing page, e.g.:
#   "gemini-3.5-flash": (<input_price>, <output_price>),
PRICING: dict[str, tuple[float, float]] = {}

CACHE_HIT_STATUSES = ("hit", "semantic", "coalesced")

metadata = MetaData()
request_log = Table(
    "request_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False),
    Column("key_id", String(32), nullable=False),
    Column("provider", String(32)),
    Column("model", String(128), nullable=False),
    Column("cache_status", String(16), nullable=False),  # miss|hit|semantic|coalesced|none
    Column("status", Integer, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("ttft_ms", Float),  # time to first token (streaming only)
    Column("prompt_tokens", Integer, nullable=False, default=0),
    Column("completion_tokens", Integer, nullable=False, default=0),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("attempts", Integer, nullable=False, default=0),
    Column("stream", Boolean, nullable=False, default=False),
    Index("ix_request_log_ts", "ts"),
)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


def normalize_url(url: str) -> str:
    """Accept the `postgres://` / `postgresql://` URLs that hosting platforms hand out."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted list."""
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil(pct / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


class RequestLog:
    def __init__(self, url: str, batch_size: int = 500, queue_max: int = 50_000):
        url = normalize_url(url)
        kwargs: dict = {"pool_pre_ping": True}
        is_sqlite = url.startswith("sqlite")
        if is_sqlite:
            kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        self.engine = create_engine(url, **kwargs)
        if is_sqlite:
            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _):  # WAL = readers don't block the batch writer
                dbapi_conn.execute("PRAGMA journal_mode=WAL")
                dbapi_conn.execute("PRAGMA synchronous=NORMAL")
        metadata.create_all(self.engine)

        self.dialect = self.engine.dialect.name
        self.batch_size = batch_size
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=queue_max)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="request-log-writer", daemon=True)
        self._thread.start()

    # -- writing ---------------------------------------------------------------
    def record(
        self, *, key_id: str, provider: Optional[str], model: str, cache_status: str,
        status: int, latency_ms: float, ttft_ms: Optional[float] = None,
        prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0,
        attempts: int = 0, stream: bool = False,
    ) -> None:
        row = dict(
            ts=time.time(), key_id=key_id, provider=provider, model=model,
            cache_status=cache_status, status=status, latency_ms=latency_ms,
            ttft_ms=ttft_ms, prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, cost_usd=cost_usd,
            attempts=attempts, stream=stream,
        )
        try:
            self._q.put_nowait(row)
        except queue.Full:  # shed load rather than slow down requests
            self.dropped += 1

    def _run(self) -> None:
        while True:
            try:
                first = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            try:
                with self.engine.begin() as conn:
                    conn.execute(insert(request_log), batch)
            except Exception:
                log.exception("failed to write %d request-log rows", len(batch))
            finally:
                for _ in batch:
                    self._q.task_done()

    def flush(self) -> None:
        """Block until every queued row has been written."""
        self._q.join()

    def close(self) -> None:
        self.flush()
        self._stop.set()
        self._thread.join(timeout=2)
        self.engine.dispose()

    # -- reading ---------------------------------------------------------------
    def stats(self, window_minutes: Optional[int] = None) -> dict:
        self.flush()
        t = request_log.c
        since = time.time() - window_minutes * 60 if window_minutes else 0.0
        in_window = t.ts >= since
        with self.engine.connect() as conn:
            total, tokens, cost, retried, errors = conn.execute(
                select(
                    func.count(),
                    func.coalesce(func.sum(t.prompt_tokens + t.completion_tokens), 0),
                    func.coalesce(func.sum(t.cost_usd), 0.0),
                    func.coalesce(func.sum(case((t.attempts > 1, 1), else_=0)), 0),
                    func.coalesce(func.sum(case((t.status >= 400, 1), else_=0)), 0),
                ).where(in_window)
            ).one()
            if total == 0:
                return self._empty()
            hits = conn.execute(
                select(func.count()).where(in_window, t.cache_status.in_(CACHE_HIT_STATUSES))
            ).scalar_one()
            by_provider = dict(conn.execute(
                select(t.provider, func.count()).where(in_window, t.provider.is_not(None))
                .group_by(t.provider)
            ).all())
            by_model = dict(conn.execute(
                select(t.model, func.count()).where(in_window).group_by(t.model)
            ).all())
            by_status = dict(conn.execute(
                select(t.status, func.count()).where(in_window).group_by(t.status)
            ).all())
            rows = conn.execute(
                select(t.latency_ms, t.cache_status, t.ttft_ms)
                .where(in_window, t.status < 400).order_by(t.ts.desc()).limit(100_000)
            ).all()

        lat = sorted(r[0] for r in rows)
        hit_lat = sorted(r[0] for r in rows if r[1] in ("hit", "semantic"))  # true cache reads
        miss_lat = sorted(r[0] for r in rows if r[1] == "miss")              # went upstream
        ttft = sorted(r[2] for r in rows if r[2] is not None)
        return {
            "total_requests": total,
            "cache_hit_rate": round(hits / total, 4),
            "error_rate": round(errors / total, 4),
            "retried_requests": retried,
            "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
            "p50_latency_ms": round(percentile(lat, 50), 2),
            "p95_latency_ms": round(percentile(lat, 95), 2),
            "p99_latency_ms": round(percentile(lat, 99), 2),
            "p95_cache_hit_latency_ms": round(percentile(hit_lat, 95), 2),
            "p95_cache_miss_latency_ms": round(percentile(miss_lat, 95), 2),
            "p50_time_to_first_token_ms": round(percentile(ttft, 50), 2),
            "total_tokens": int(tokens),
            "total_cost_usd": round(float(cost), 6),
            "by_provider": by_provider,
            "by_model": by_model,
            "by_status": {str(k): v for k, v in by_status.items()},
            "log_rows_dropped": self.dropped,
        }

    @staticmethod
    def _empty() -> dict:
        return {
            "total_requests": 0, "cache_hit_rate": 0.0, "error_rate": 0.0,
            "retried_requests": 0, "avg_latency_ms": 0.0, "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0, "p99_latency_ms": 0.0,
            "p95_cache_hit_latency_ms": 0.0, "p95_cache_miss_latency_ms": 0.0,
            "p50_time_to_first_token_ms": 0.0, "total_tokens": 0, "total_cost_usd": 0.0,
            "by_provider": {}, "by_model": {}, "by_status": {}, "log_rows_dropped": 0,
        }

    def timeseries(self, minutes: int = 30) -> list[dict]:
        """Per-minute request volume, errors, cache hits and latency for charts."""
        self.flush()
        t = request_log.c
        since = time.time() - minutes * 60
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(t.ts, t.latency_ms, t.cache_status, t.status).where(t.ts >= since)
            ).all()
        buckets: dict[int, dict] = {}
        for ts, latency, cache_status, status in rows:
            b = buckets.setdefault(
                int(ts // 60) * 60, {"requests": 0, "errors": 0, "cache_hits": 0, "lat": []}
            )
            b["requests"] += 1
            b["errors"] += status >= 400
            b["cache_hits"] += cache_status in CACHE_HIT_STATUSES
            if status < 400:
                b["lat"].append(latency)
        out = []
        for minute in sorted(buckets):
            b = buckets[minute]
            lat = sorted(b["lat"])
            out.append({
                "t": minute, "requests": b["requests"], "errors": b["errors"],
                "cache_hits": b["cache_hits"],
                "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
                "p95_latency_ms": round(percentile(lat, 95), 2),
            })
        return out
