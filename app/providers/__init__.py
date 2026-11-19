from app.config import Settings
from app.providers.base import Provider, ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider

__all__ = ["Provider", "ProviderError", "GeminiProvider", "MockProvider", "build_providers"]


def build_providers(settings: Settings) -> list[Provider]:
    """Instantiate providers in the configured fallback order."""
    providers: list[Provider] = []
    for name in settings.provider_order:
        if name == "gemini":
            if settings.gemini_api_key:
                providers.append(GeminiProvider(settings.gemini_api_key))
        elif name == "mock":
            providers.append(MockProvider())
    if not providers:
        providers.append(MockProvider())
    return providers
