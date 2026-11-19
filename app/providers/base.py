from abc import ABC, abstractmethod

from app.schemas import ChatRequest, ChatResponse


class ProviderError(Exception):
    """Raised when an upstream provider fails; the router will try the next one."""


class Provider(ABC):
    name: str

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        ...
