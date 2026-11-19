"""Request log + usage stats, backed by SQLite (Postgres planned)."""
import math
import sqlite3
import threading
import time

# Approximate USD per 1M tokens (input, output). Override as pricing changes.
PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000


class RequestLog:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    key_id TEXT NOT NULL,
                    provider TEXT,
                    model TEXT NOT NULL,
                    cached INTEGER NOT NULL,
                    status INTEGER NOT NULL,
                    latency_ms REAL NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0
                )"""
            )
            self._conn.commit()

    def record(self, *, key_id, provider, model, cached, status, latency_ms,
               prompt_tokens=0, completion_tokens=0, cost_usd=0.0) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO requests
                   (ts, key_id, provider, model, cached, status, latency_ms,
                    prompt_tokens, completion_tokens, cost_usd)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (time.time(), key_id, provider, model, int(cached), status,
                 latency_ms, prompt_tokens, completion_tokens, cost_usd),
            )
            self._conn.commit()

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT provider, cached, status, latency_ms, prompt_tokens, "
                "completion_tokens, cost_usd FROM requests"
            ).fetchall()
        total = len(rows)
        if total == 0:
            return {"total_requests": 0, "cache_hit_rate": 0.0, "error_rate": 0.0,
                    "avg_latency_ms": 0.0, "p95_latency_ms": 0.0,
                    "total_tokens": 0, "total_cost_usd": 0.0, "by_provider": {}}
        latencies = sorted(r[3] for r in rows)
        p95 = latencies[max(0, math.ceil(0.95 * total) - 1)]
        by_provider: dict[str, int] = {}
        for provider, cached, *_ in rows:
            if provider and not cached:
                by_provider[provider] = by_provider.get(provider, 0) + 1
        return {
            "total_requests": total,
            "cache_hit_rate": round(sum(r[1] for r in rows) / total, 4),
            "error_rate": round(sum(1 for r in rows if r[2] >= 400) / total, 4),
            "avg_latency_ms": round(sum(latencies) / total, 2),
            "p95_latency_ms": round(p95, 2),
            "total_tokens": sum(r[4] + r[5] for r in rows),
            "total_cost_usd": round(sum(r[6] for r in rows), 6),
            "by_provider": by_provider,
        }
