from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from app.schemas import ChatRequest, ChatResponse, Usage

# HTTP statuses worth retrying: timeouts, rate limits, upstream overload.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class ProviderError(Exception):
    """An upstream provider failed.

    `retryable` marks transient failures (timeouts, 429, 5xx). Permanent
    failures (bad model name, auth, safety block) skip retries and go straight
    to the next provider.
    """

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def status_error(provider: str, status: int, body: str) -> ProviderError:
    return ProviderError(
        f"{provider} returned HTTP {status}: {body[:200]}",
        retryable=status in RETRYABLE_STATUS,
    )


@dataclass
class StreamEvent:
    """One piece of a streamed completion: a text delta and/or final usage."""

    delta: str = ""
    usage: Optional[Usage] = None


class Provider(ABC):
    name: str
    model_override: Optional[str] = None

    def resolve_model(self, request: ChatRequest) -> str:
        """Model names differ per provider, so a provider may pin its own."""
        return self.model_override or request.model

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        ...

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Default: no native streaming, so emit the whole reply as one event."""
        response = await self.chat(request)
        yield StreamEvent(delta=response.choices[0].message.content)
        yield StreamEvent(usage=response.usage)

    async def aclose(self) -> None:
        return None
