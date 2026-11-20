import hashlib
import time
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.responses import RedirectResponse

from app.cache import ResponseCache, cache_key, is_cacheable
from app.config import Settings
from app.db import RequestLog, estimate_cost
from app.providers import Provider, build_providers
from app.ratelimit import TokenBucketLimiter
from app.router import AllProvidersFailed, route_with_fallback
from app.schemas import ChatRequest, ChatResponse


def create_app(
    settings: Optional[Settings] = None,
    providers: Optional[list[Provider]] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    providers = providers if providers is not None else build_providers(settings)

    app = FastAPI(title="LLM Gateway", version="0.1.0")
    limiter = TokenBucketLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)
    cache = ResponseCache(settings.cache_ttl_seconds, settings.cache_max_entries)
    log = RequestLog(settings.db_path)

    def authenticate(authorization: Optional[str] = Header(default=None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        if token not in settings.api_keys:
            raise HTTPException(401, "Invalid API key")
        return hashlib.sha256(token.encode()).hexdigest()[:12]  # never log raw keys

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "providers": [p.name for p in providers]}

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def chat_completions(
        request: ChatRequest, response: Response, key_id: str = Depends(authenticate)
    ):
        start = time.perf_counter()

        allowed, retry_after = limiter.acquire(key_id)
        if not allowed:
            log.record(key_id=key_id, provider=None, model=request.model, cached=False,
                       status=429, latency_ms=(time.perf_counter() - start) * 1000)
            raise HTTPException(
                429, "Rate limit exceeded",
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )

        key = cache_key(request) if is_cacheable(request) else None
        if key and (hit := cache.get(key)):
            response.headers["X-Cache"] = "HIT"
            log.record(key_id=key_id, provider="cache", model=request.model, cached=True,
                       status=200, latency_ms=(time.perf_counter() - start) * 1000)
            return hit

        try:
            provider_name, result = await route_with_fallback(providers, request)
        except AllProvidersFailed as exc:
            log.record(key_id=key_id, provider=None, model=request.model, cached=False,
                       status=502, latency_ms=(time.perf_counter() - start) * 1000)
            raise HTTPException(502, f"All providers failed: {exc}")

        if key:
            cache.set(key, result)
        cost = estimate_cost(
            request.model, result.usage.prompt_tokens, result.usage.completion_tokens
        )
        log.record(
            key_id=key_id, provider=provider_name, model=request.model, cached=False,
            status=200, latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens, cost_usd=cost,
        )
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Gateway-Provider"] = provider_name
        return result

    @app.get("/v1/stats")
    def stats(_: str = Depends(authenticate)) -> dict:
        return log.stats()

    return app


app = create_app()
