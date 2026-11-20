import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers import MockProvider, Provider, ProviderError
from app.ratelimit import TokenBucketLimiter

AUTH = {"Authorization": "Bearer test-key"}
BODY = {"model": "gemini-3.5-flash", "messages": [{"role": "user", "content": "hi"}]}


class FailingProvider(Provider):
    name = "broken"

    async def chat(self, request):
        raise ProviderError("boom")


def make_client(tmp_path, providers=None, **overrides):
    settings = Settings(api_keys=["test-key"], db_path=str(tmp_path / "t.db"), **overrides)
    return TestClient(create_app(settings, providers or [MockProvider()]))


def test_health(tmp_path):
    assert make_client(tmp_path).get("/health").json()["status"] == "ok"


def test_requires_auth(tmp_path):
    c = make_client(tmp_path)
    assert c.post("/v1/chat/completions", json=BODY).status_code == 401
    bad = {"Authorization": "Bearer nope"}
    assert c.post("/v1/chat/completions", json=BODY, headers=bad).status_code == 401


def test_openai_shaped_response(tmp_path):
    r = make_client(tmp_path).post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"].startswith("[mock]")
    assert data["usage"]["total_tokens"] > 0


def test_cache_hit_on_repeat(tmp_path):
    c = make_client(tmp_path)
    first = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    second = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "HIT"
    assert second.json()["id"] == first.json()["id"]


def test_nondeterministic_requests_not_cached(tmp_path):
    c = make_client(tmp_path)
    body = {**BODY, "temperature": 0.9}
    c.post("/v1/chat/completions", json=body, headers=AUTH)
    assert c.post("/v1/chat/completions", json=body, headers=AUTH).headers["X-Cache"] == "MISS"


def test_fallback_to_next_provider(tmp_path):
    c = make_client(tmp_path, providers=[FailingProvider(), MockProvider()])
    r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["X-Gateway-Provider"] == "mock"


def test_all_providers_fail_returns_502(tmp_path):
    c = make_client(tmp_path, providers=[FailingProvider()])
    assert c.post("/v1/chat/completions", json=BODY, headers=AUTH).status_code == 502


def test_rate_limit_returns_429(tmp_path):
    c = make_client(tmp_path, rate_limit_rpm=1, rate_limit_burst=2)
    codes = [c.post("/v1/chat/completions", json=BODY, headers=AUTH).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert "Retry-After" in r.headers


def test_stats(tmp_path):
    c = make_client(tmp_path)
    c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    s = c.get("/v1/stats", headers=AUTH).json()
    assert s["total_requests"] == 2
    assert s["cache_hit_rate"] == 0.5
    assert s["by_provider"] == {"mock": 1}


def test_token_bucket_refills_over_time():
    now = [0.0]
    limiter = TokenBucketLimiter(rpm=60, burst=1, clock=lambda: now[0])
    assert limiter.acquire("k")[0] is True
    assert limiter.acquire("k")[0] is False
    now[0] += 1.0  # 1 token/sec refill
    assert limiter.acquire("k")[0] is True
