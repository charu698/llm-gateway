import asyncio

import pytest

from app.providers import MockProvider, Provider, ProviderError, StreamEvent
from app.resilience import CircuitBreaker, RetryPolicy
from app.router import AllProvidersFailed, Router
from app.schemas import ChatRequest, Message
from tests.conftest import FailingProvider

REQ = ChatRequest(model="m", messages=[Message(role="user", content="hi")])


class FlakyThenOk(MockProvider):
    """Fails `failures` times (retryable), then behaves."""

    def __init__(self, failures: int, name="flaky"):
        super().__init__(name=name)
        self.failures = failures
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        if self.calls <= self.failures:
            raise ProviderError("503", retryable=True)
        return await super().chat(request)


def make_router(providers, max_attempts=3, threshold=5, clock=None, sleeps=None):
    sleeps = sleeps if sleeps is not None else []

    async def fake_sleep(s):
        sleeps.append(s)

    breakers = {
        p.name: CircuitBreaker(threshold, 30.0, **({"clock": clock} if clock else {}))
        for p in providers
    }
    return Router(
        providers, RetryPolicy(max_attempts, 0.2, 2.0), breakers, sleep=fake_sleep, rand=lambda: 0.0
    ), sleeps


def test_retry_policy_exponential_backoff_with_cap_and_jitter():
    p = RetryPolicy(max_attempts=5, base_delay=0.2, max_delay=1.0, jitter=0.25)
    assert [p.delay(n, rand=lambda: 0.0) for n in (1, 2, 3, 4)] == [0.2, 0.4, 0.8, 1.0]
    assert p.delay(1, rand=lambda: 1.0) == pytest.approx(0.25)  # +25% jitter


def test_router_retries_then_succeeds_with_backoff():
    provider = FlakyThenOk(failures=2)
    router, sleeps = make_router([provider])
    result = asyncio.run(router.chat(REQ))
    assert result.provider == "flaky" and result.attempts == 3
    assert sleeps == [0.2, 0.4]  # exponential backoff between attempts


def test_non_retryable_error_skips_retries_and_falls_back():
    bad = FailingProvider("bad", retryable=False)
    router, sleeps = make_router([bad, MockProvider()])
    result = asyncio.run(router.chat(REQ))
    assert bad.calls == 1 and sleeps == []
    assert result.provider == "mock"


def test_gives_up_after_max_attempts():
    bad = FailingProvider("bad")
    router, _ = make_router([bad], max_attempts=3)
    with pytest.raises(AllProvidersFailed):
        asyncio.run(router.chat(REQ))
    assert bad.calls == 3


def test_circuit_breaker_state_machine():
    now = [0.0]
    cb = CircuitBreaker(failure_threshold=3, recovery_seconds=10, clock=lambda: now[0])
    assert cb.state == "closed" and cb.allow()
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open" and not cb.allow()
    now[0] = 10.0
    assert cb.state == "half_open"
    assert cb.allow() is True       # one probe
    assert cb.allow() is False      # but only one
    cb.record_failure()             # probe fails -> reopen
    assert cb.state == "open"
    now[0] = 20.0
    assert cb.allow() is True
    cb.record_success()             # probe succeeds -> closed
    assert cb.state == "closed" and cb.allow()


def test_open_circuit_skips_provider_until_recovery():
    now = [0.0]
    bad = FailingProvider("bad")
    router, _ = make_router([bad, MockProvider()], max_attempts=1, threshold=3, clock=lambda: now[0])
    for _ in range(3):  # three requests trip the breaker
        asyncio.run(router.chat(REQ))
    assert bad.calls == 3
    asyncio.run(router.chat(REQ))  # circuit open: bad provider not even called
    assert bad.calls == 3
    assert router.states()[0]["circuit"] == "open"
    now[0] = 31.0                  # after recovery window a probe is allowed
    asyncio.run(router.chat(REQ))
    assert bad.calls == 4


def test_all_circuits_open_fails_fast():
    now = [0.0]
    bad = FailingProvider("bad")
    router, _ = make_router([bad], max_attempts=1, threshold=1, clock=lambda: now[0])
    with pytest.raises(AllProvidersFailed):
        asyncio.run(router.chat(REQ))
    with pytest.raises(AllProvidersFailed) as exc:
        asyncio.run(router.chat(REQ))
    assert "circuit open" in str(exc.value) and bad.calls == 1


class StreamFailsFirst(Provider):
    name = "sfail"

    async def chat(self, request):
        raise NotImplementedError

    async def stream(self, request):
        raise ProviderError("503", retryable=False)
        yield  # pragma: no cover  (makes this an async generator)


class StreamBreaksMidway(Provider):
    name = "midway"

    async def chat(self, request):
        raise NotImplementedError

    async def stream(self, request):
        yield StreamEvent(delta="partial ")
        raise ProviderError("connection reset", retryable=True)


def test_stream_fails_over_before_first_event():
    router, _ = make_router([StreamFailsFirst(), MockProvider()])

    async def go():
        res = await router.stream(REQ)
        return res.provider, "".join([e.delta async for e in res.events])

    provider, text = asyncio.run(go())
    assert provider == "mock" and text.startswith("[mock]")


def test_stream_failure_after_first_event_is_surfaced_not_retried():
    router, _ = make_router([StreamBreaksMidway(), MockProvider()])

    async def go():
        res = await router.stream(REQ)
        got = []
        with pytest.raises(ProviderError):
            async for e in res.events:
                got.append(e.delta)
        return res.provider, got

    provider, got = asyncio.run(go())
    assert provider == "midway" and got == ["partial "]
