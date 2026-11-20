from app.config import Settings
from app.providers.base import Provider, ProviderError, StreamEvent
from app.providers.gemini import GeminiProvider
from app.providers.mock import MockProvider
from app.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "Provider",
    "ProviderError",
    "StreamEvent",
    "GeminiProvider",
    "OpenAICompatProvider",
    "MockProvider",
    "build_providers",
]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_providers(settings: Settings) -> list[Provider]:
    """Instantiate providers in the configured fallback order.

    Providers whose API key is missing are skipped; if nothing is left the
    offline mock provider is used so the gateway always starts.
    """
    t = settings.provider_timeout
    providers: list[Provider] = []
    for name in settings.provider_order:
        if name == "gemini" and settings.gemini_api_key:
            providers.append(
                GeminiProvider(
                    settings.gemini_api_key, timeout=t, model_override=settings.gemini_model
                )
            )
        elif name == "groq" and settings.groq_api_key:
            providers.append(
                OpenAICompatProvider(
                    "groq", GROQ_BASE_URL, settings.groq_api_key,
                    model_override=settings.groq_model, timeout=t,
                )
            )
        elif name == "openrouter" and settings.openrouter_api_key:
            providers.append(
                OpenAICompatProvider(
                    "openrouter", OPENROUTER_BASE_URL, settings.openrouter_api_key,
                    model_override=settings.openrouter_model, timeout=t,
                )
            )
        elif name == "mock":
            providers.append(MockProvider("mock", latency_ms=settings.mock_latency_ms))
        elif name == "flaky":
            providers.append(
                MockProvider(
                    "flaky",
                    latency_ms=settings.mock_latency_ms,
                    fail_rate=settings.flaky_fail_rate,
                )
            )
    if not providers:
        providers.append(MockProvider("mock", latency_ms=settings.mock_latency_ms))
    return providers
