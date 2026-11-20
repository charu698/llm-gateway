import asyncio
import random
import re
from typing import AsyncIterator, Callable

from app.providers.base import Provider, ProviderError, StreamEvent
from app.schemas import ChatRequest, ChatResponse, Choice, ChoiceMessage, Usage


class MockProvider(Provider):
    """Deterministic offline provider for dev, tests, demos and benchmarks.

    `latency_ms` simulates upstream LLM latency; `fail_rate` makes a fraction of
    calls fail with a retryable error (used to benchmark retries and failover).
    """

    def __init__(
        self,
        name: str = "mock",
        latency_ms: float = 0,
        fail_rate: float = 0.0,
        rng: Callable[[], float] = random.random,
    ):
        self.name = name
        self.latency = latency_ms / 1000.0
        self.fail_rate = fail_rate
        self._rng = rng

    def _maybe_fail(self) -> None:
        if self.fail_rate and self._rng() < self.fail_rate:
            raise ProviderError(f"{self.name}: simulated upstream failure (503)", retryable=True)

    @staticmethod
    def _reply(request: ChatRequest) -> str:
        last_user = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), ""
        )
        return f"[mock] You said: {last_user}"

    @staticmethod
    def _usage(request: ChatRequest, reply: str) -> Usage:
        prompt = sum(len(m.content.split()) for m in request.messages)
        completion = len(reply.split())
        return Usage(
            prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        self._maybe_fail()
        if self.latency:
            await asyncio.sleep(self.latency)
        reply = self._reply(request)
        return ChatResponse(
            model=self.resolve_model(request),
            choices=[Choice(message=ChoiceMessage(content=reply))],
            usage=self._usage(request, reply),
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self._maybe_fail()
        reply = self._reply(request)
        pieces = re.findall(r"\S+\s*", reply)
        per_piece = self.latency / max(1, len(pieces))
        for piece in pieces:
            if per_piece:
                await asyncio.sleep(per_piece)
            yield StreamEvent(delta=piece)
        yield StreamEvent(usage=self._usage(request, reply))
