import asyncio

import httpx

from app.cache import SingleFlight
from app.config import Settings
from app.main import create_app
from tests.conftest import AUTH, BODY, CountingProvider, make_settings


def test_singleflight_runs_function_once_for_concurrent_callers():
    calls = []

    async def work():
        calls.append(1)
        await asyncio.sleep(0.05)
        return "result"

    async def go():
        sf = SingleFlight()
        return await asyncio.gather(*(sf.run("k", work) for _ in range(10)))

    results = asyncio.run(go())
    assert len(calls) == 1
    assert all(r[0] == "result" for r in results)
    assert sorted(r[1] for r in results) == [False] + [True] * 9  # 1 leader, 9 followers


def test_singleflight_propagates_errors_to_followers():
    async def boom():
        await asyncio.sleep(0.02)
        raise ValueError("upstream down")

    async def go():
        sf = SingleFlight()
        return await asyncio.gather(*(sf.run("k", boom) for _ in range(3)), return_exceptions=True)

    assert all(isinstance(r, ValueError) for r in asyncio.run(go()))


def test_singleflight_follower_takes_over_if_leader_cancelled():
    async def go():
        sf = SingleFlight()
        started = asyncio.Event()

        async def slow():
            started.set()
            await asyncio.sleep(10)

        async def quick():
            return "recovered"

        leader = asyncio.create_task(sf.run("k", slow))
        await started.wait()
        follower = asyncio.create_task(sf.run("k", quick))
        await asyncio.sleep(0.01)
        leader.cancel()
        return await follower

    assert asyncio.run(go()) == ("recovered", False)


def test_gateway_collapses_concurrent_identical_requests(tmp_path):
    provider = CountingProvider(latency_ms=100)
    app = create_app(make_settings(tmp_path, rate_limit_burst=1000), [provider])

    async def go():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as client:
            rs = await asyncio.gather(*(
                client.post("/v1/chat/completions", json=BODY, headers=AUTH) for _ in range(20)
            ))
            return rs

    responses = asyncio.run(go())
    assert all(r.status_code == 200 for r in responses)
    assert provider.calls == 1  # 20 identical in-flight requests -> one upstream call
    kinds = sorted(r.headers["X-Cache"] for r in responses)
    assert kinds.count("MISS") == 1 and kinds.count("COALESCED") == 19
