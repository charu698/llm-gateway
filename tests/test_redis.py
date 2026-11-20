"""Runs against a real Redis (set REDIS_URL, e.g. redis://localhost:6379/15)."""
import asyncio
import time

import redis
import redis.asyncio as aioredis

from app.cache import RedisCache, cache_key
from app.ratelimit import RedisTokenBucketLimiter
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Message, Usage
from tests.conftest import AUTH, BODY, REDIS_URL, CountingProvider, make_client, needs_redis

pytestmark = needs_redis


def fresh_redis():
    r = redis.Redis.from_url(REDIS_URL)
    r.flushdb()
    return r


def test_redis_limiter_burst_then_deny_then_refill():
    fresh_redis()

    async def go():
        client = aioredis.from_url(REDIS_URL)
        limiter = RedisTokenBucketLimiter(client, rpm=120, burst=3)  # 2 tokens/sec
        allowed = [(await limiter.acquire("k"))[0] for _ in range(5)]
        retry_after = (await limiter.acquire("k"))[1]
        await asyncio.sleep(0.6)  # ~1.2 tokens refilled
        refilled = (await limiter.acquire("k"))[0]
        other_key = (await limiter.acquire("other"))[0]
        await client.aclose()
        return allowed, retry_after, refilled, other_key

    allowed, retry_after, refilled, other_key = asyncio.run(go())
    assert allowed == [True, True, True, False, False]
    assert 0 < retry_after <= 0.5
    assert refilled is True and other_key is True


def test_redis_limiter_is_atomic_under_concurrency():
    fresh_redis()

    async def go():
        client = aioredis.from_url(REDIS_URL)
        limiter = RedisTokenBucketLimiter(client, rpm=1, burst=10)
        results = await asyncio.gather(*(limiter.acquire("k") for _ in range(100)))
        await client.aclose()
        return sum(1 for ok, _ in results if ok)

    assert asyncio.run(go()) == 10  # exactly `burst`, never more


def test_redis_limiter_fails_open_when_redis_is_down():
    async def go():
        client = aioredis.from_url("redis://localhost:1")  # nothing listens here
        limiter = RedisTokenBucketLimiter(client, rpm=1, burst=1)
        out = await limiter.acquire("k")
        await client.aclose()
        return out

    assert asyncio.run(go()) == (True, 0.0)


def test_redis_cache_roundtrip_and_ttl():
    fresh_redis()
    resp = ChatResponse(
        model="m", choices=[Choice(message=ChoiceMessage(content="hello"))], usage=Usage()
    )
    key = cache_key(ChatRequest(model="m", messages=[Message(role="user", content="x")]))

    async def go():
        client = aioredis.from_url(REDIS_URL)
        cache = RedisCache(client, ttl=1)
        assert await cache.get(key) is None
        await cache.set(key, resp)
        got = await cache.get(key)
        await asyncio.sleep(1.2)
        expired = await cache.get(key)
        await client.aclose()
        return got, expired

    got, expired = asyncio.run(go())
    assert got.choices[0].message.content == "hello" and got.id == resp.id
    assert expired is None


def test_two_gateway_instances_share_cache_and_rate_limit(tmp_path):
    fresh_redis()
    p1, p2 = CountingProvider(), CountingProvider()
    kw = dict(redis_url=REDIS_URL, rate_limit_rpm=1, rate_limit_burst=4)
    with make_client(tmp_path, providers=[p1], **kw) as a, \
         make_client(tmp_path, providers=[p2], database_url=f"sqlite:///{tmp_path / 'b.db'}", **kw) as b:
        assert a.get("/health").json()["cache"] == "redis"
        first = a.post("/v1/chat/completions", json=BODY, headers=AUTH)
        second = b.post("/v1/chat/completions", json=BODY, headers=AUTH)  # other instance
        # burst=4 shared across both instances: 2 used so far, 2 more allowed, then 429
        more = [(a if i % 2 else b).post("/v1/chat/completions", json=BODY, headers=AUTH).status_code
                for i in range(3)]
    assert first.headers["X-Cache"] == "MISS" and second.headers["X-Cache"] == "HIT"
    assert p1.calls == 1 and p2.calls == 0
    assert more == [200, 200, 429]


def test_gateway_survives_redis_outage(tmp_path):
    provider = CountingProvider()
    with make_client(tmp_path, providers=[provider], redis_url="redis://localhost:1") as c:
        r = c.post("/v1/chat/completions", json=BODY, headers=AUTH)
    assert r.status_code == 200 and provider.calls == 1  # fails open, still serves
