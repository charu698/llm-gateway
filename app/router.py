import logging

from app.providers import Provider, ProviderError
from app.schemas import ChatRequest, ChatResponse

log = logging.getLogger("gateway.router")


class AllProvidersFailed(Exception):
    pass


async def route_with_fallback(
    providers: list[Provider], request: ChatRequest
) -> tuple[str, ChatResponse]:
    """Try providers in order; return (provider_name, response) from the first that works."""
    errors: list[str] = []
    for provider in providers:
        try:
            return provider.name, await provider.chat(request)
        except ProviderError as exc:
            log.warning("provider %s failed, falling back: %s", provider.name, exc)
            errors.append(str(exc))
    raise AllProvidersFailed("; ".join(errors) or "no providers configured")
