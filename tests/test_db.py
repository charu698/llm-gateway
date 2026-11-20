import time

from app.db import RequestLog, normalize_url, percentile
from tests.conftest import TEST_DATABASE_URL, needs_postgres


def test_percentile_nearest_rank():
    values = list(range(1, 101))
    assert percentile(values, 50) == 50
    assert percentile(values, 95) == 95
    assert percentile(values, 99) == 99
    assert percentile([], 95) == 0.0


def test_normalize_url_handles_platform_postgres_urls():
    assert normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def exercise(url: str):
    log = RequestLog(url)
    for i in range(100):
        hit = i % 4 == 0
        log.record(
            key_id="k", provider="cache" if hit else "mock", model="m",
            cache_status="hit" if hit else "miss", status=200 if i < 98 else 502,
            latency_ms=float(i + 1), prompt_tokens=10, completion_tokens=5,
            cost_usd=0.001, attempts=2 if i == 5 else 1,
        )
    s = log.stats()
    series = log.timeseries(5)
    window = log.stats(window_minutes=1)
    log.close()
    return s, series, window


def check(s, series, window):
    assert s["total_requests"] == 100
    assert s["cache_hit_rate"] == 0.25
    assert s["error_rate"] == 0.02
    assert s["retried_requests"] == 1
    assert s["total_tokens"] == 1500
    assert abs(s["total_cost_usd"] - 0.1) < 1e-9
    assert s["by_provider"] == {"cache": 25, "mock": 75}
    assert s["by_status"] == {"200": 98, "502": 2}
    assert s["p50_latency_ms"] == 49.0 or s["p50_latency_ms"] == 50.0
    assert s["p95_cache_hit_latency_ms"] < s["p95_cache_miss_latency_ms"] + 100
    assert sum(b["requests"] for b in series) == 100
    assert window["total_requests"] == 100


def test_request_log_on_sqlite(tmp_path):
    check(*exercise(f"sqlite:///{tmp_path / 'log.db'}"))


def test_empty_stats(tmp_path):
    log = RequestLog(f"sqlite:///{tmp_path / 'e.db'}")
    assert log.stats()["total_requests"] == 0 and log.timeseries() == []
    log.close()


def test_batched_writes_do_not_lose_rows(tmp_path):
    log = RequestLog(f"sqlite:///{tmp_path / 'b.db'}", batch_size=50)
    for _ in range(2000):
        log.record(key_id="k", provider="mock", model="m", cache_status="miss",
                   status=200, latency_ms=1.0)
    assert log.stats()["total_requests"] == 2000 and log.dropped == 0
    log.close()


@needs_postgres
def test_request_log_on_postgres():
    from sqlalchemy import create_engine, text

    engine = create_engine(normalize_url(TEST_DATABASE_URL))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS request_log"))
    check(*exercise(TEST_DATABASE_URL))
