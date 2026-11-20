"""Provider routing: retries with backoff, circuit breakers, ordered failover."""
import asyncio
import logging
import random
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

from app.providers import Provider, ProviderError, StreamEvent
from app.resilience import CircuitBreaker, RetryPolicy
from app.schemas import ChatRequest, ChatResponse

log = logging.getLogger("gateway.router")


class AllProvidersFailed(Exception):
    pass


@dataclass
class RouteResult:
    provider: str
    response: ChatResponse
    attempts: int


@dataclass
class StreamResult:
    provider: str
    events: AsyncIterator[StreamEvent]
    attempts: int


class Router:
    def __init__(
        self,
        providers: list[Provider],
        retry: RetryPolicy,
        breakers: dict[str, CircuitBreaker],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rand: Callable[[], float] = random.random,
    ):
        self.providers = providers
        self.retry = retry
        self.breakers = breakers
        self._sleep = sleep
        self._rand = rand

    def states(self) -> list[dict]:
        return [{"name": p.name, "circuit": self.breakers[p.name].state} for p in self.providers]

    async def _attempt_loop(
        self,
        provider: Provider,
        call: Callable[[], Awaitable],
        errors: list[str],
        counter: list[int],
    ):
        """Run `call` with retries. Returns its result, or None if this provider gave up."""
        breaker = self.breakers[provider.name]
        for attempt in range(1, self.retry.max_attempts + 1):
            counter[0] += 1
            try:
                return await call()
            except ProviderError as exc:
                breaker.record_failure()
                errors.append(f"{provider.name}: {exc}")
                log.warning(
                    "provider=%s attempt=%d/%d failed (retryable=%s): %s",
                    provider.name, attempt, self.retry.max_attempts, exc.retryable, exc,
                )
                out_of_tries = attempt == self.retry.max_attempts
                if not exc.retryable or out_of_tries or breaker.state != CircuitBreaker.CLOSED:
                    return None
                await self._sleep(self.retry.delay(attempt, self._rand))
        return None

    async def chat(self, request: ChatRequest) -> RouteResult:
        errors: list[str] = []
        counter = [0]
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                errors.append(f"{provider.name}: circuit open, skipped")
                continue
            response = await self._attempt_loop(
                provider, lambda p=provider: p.chat(request), errors, counter
            )
            if response is not None:
                breaker.record_success()
                return RouteResult(provider.name, response, counter[0])
        raise AllProvidersFailed("; ".join(errors) or "no providers configured")

    async def stream(self, request: ChatRequest) -> StreamResult:
        """Failover works until the first event arrives; after that the stream is committed."""
        errors: list[str] = []
        counter = [0]
        for provider in self.providers:
            breaker = self.breakers[provider.name]
            if not breaker.allow():
                errors.append(f"{provider.name}: circuit open, skipped")
                continue

            async def open_stream(p=provider):
                agen = p.stream(request).__aiter__()
                try:
                    first: Optional[StreamEvent] = await agen.__anext__()
                except StopAsyncIteration:
                    first = None
                except BaseException:
                    await agen.aclose()
                    raise
                return agen, first

            opened = await self._attempt_loop(provider, open_stream, errors, counter)
            if opened is not None:
                agen, first = opened
                breaker.record_success()
                return StreamResult(
                    provider.name, self._continue(provider, agen, first), counter[0]
                )
        raise AllProvidersFailed("; ".join(errors) or "no providers configured")

    async def _continue(self, provider: Provider, agen, first: Optional[StreamEvent]):
        breaker = self.breakers[provider.name]
        try:
            if first is not None:
                yield first
            async for event in agen:
                yield event
        except ProviderError:
            breaker.record_failure()
            raise
        finally:
            await agen.aclose()
