from app.providers import MockProvider
from app.ratelimit import TokenBucketLimiter
from tests.conftest import AUTH, BODY, FailingProvider, make_client


def test_health(tmp_path):
    with make_client(tmp_path) as c:
        data = c.get("/health").json()
    assert data["status"] == "ok"
    assert data["providers"] == [{"name": "mock", "circuit": "closed"}]
    assert data["cache"] == "memory" and data["rate_limiter"] == "memory"


def test_requires_auth(tmp_path):
    with make_client(tmp_path) as c:
        assert c.post("/v1/chat/completions", json=BODY).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert c.post("/v1/chat/completions", json=BODY, headers=bad).status_code == 401
        assert c.get("/v1/stats").status_code == 401


def test_openai_shaped_response(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"].startswith("[mock]")
    assert data["usage"]["total_tokens"] > 0


def test_invalid_request_is_422(tmp_path):
    with make_client(tmp_path) as c:
        r = c.post("/v1/chat/completions", json={"model": "x", "messages": []}, headers=AUTH)
    assert r.status_code == 422


def test_cache_hit_on_repeat(tmp_path):
    with make_client(tmp_path) as c:
        first = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
        second = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["id"] == first.json()["id"]


def test_nondeterministic_requests_not_cached(tmp_path):
    body = {**BODY, "temperature": 0.9}
    with make_client(tmp_path) as c:
        c.post("/v1/chat/completions", json=body, headers=AUTH)
        r = c.post("/v1/chat/completions", json=body, headers=AUTH)
    assert r.headers["X-Cache"] == "MISS"


def test_fallback_to_next_provider(tmp_path):
    with make_client(tmp_path, providers=[FailingProvider(), MockProvider()]) as c:
        r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["X-Gateway-Provider"] == "mock"
    assert int(r.headers["X-Gateway-Attempts"]) == 4  # 3 failed tries + 1 success


def test_all_providers_fail_returns_502(tmp_path):
    with make_client(tmp_path, providers=[FailingProvider()]) as c:
        assert c.post("/v1/chat/completions", json=BODY, headers=AUTH).status_code == 502


def test_rate_limit_returns_429(tmp_path):
    with make_client(tmp_path, rate_limit_rpm=1, rate_limit_burst=2) as c:
        codes = [
            c.post("/v1/chat/completions", json=BODY, headers=AUTH).status_code for _ in range(3)
        ]
        r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert codes == [200, 200, 429]
    assert "Retry-After" in r.headers


def test_stats(tmp_path):
    with make_client(tmp_path) as c:
        c.post("/v1/chat/completions", json=BODY, headers=AUTH)
        c.post("/v1/chat/completions", json=BODY, headers=AUTH)
        s = c.get("/v1/stats", headers=AUTH).json()
    assert s["total_requests"] == 2
    assert s["cache_hit_rate"] == 0.5
    assert s["by_provider"] == {"mock": 1, "cache": 1}
    assert s["p95_latency_ms"] >= 0 and s["error_rate"] == 0


def test_stats_counts_retries_and_errors(tmp_path):
    with make_client(tmp_path, providers=[FailingProvider(), MockProvider()]) as c:
        c.post("/v1/chat/completions", json=BODY, headers=AUTH)
        s = c.get("/v1/stats", headers=AUTH).json()
    assert s["retried_requests"] == 1 and s["error_rate"] == 0


def test_timeseries_endpoint(tmp_path):
    with make_client(tmp_path) as c:
        c.post("/v1/chat/completions", json=BODY, headers=AUTH)
        series = c.get("/v1/stats/timeseries?minutes=5", headers=AUTH).json()
    assert len(series) == 1 and series[0]["requests"] == 1


def test_dashboard_and_root(tmp_path):
    with make_client(tmp_path) as c:
        assert c.get("/dashboard").status_code == 200
        assert c.get("/", follow_redirects=False).status_code in (302, 307)


def test_token_bucket_refills_over_time():
    import asyncio

    now = [0.0]
    limiter = TokenBucketLimiter(rpm=60, burst=1, clock=lambda: now[0])

    async def go():
        assert (await limiter.acquire("k"))[0] is True
        assert (await limiter.acquire("k"))[0] is False
        now[0] += 1.0  # 1 token/sec refill
        assert (await limiter.acquire("k"))[0] is True

    asyncio.run(go())
