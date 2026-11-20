import hmac
import hashlib
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from app.cache import MemoryCache, RedisCache, SingleFlight, cache_key, is_cacheable
from app.config import Settings
from app.db import RequestLog, estimate_cost
from app.providers import Provider, ProviderError, build_providers
from app.ratelimit import RedisTokenBucketLimiter, TokenBucketLimiter
from app.resilience import CircuitBreaker, RetryPolicy
from app.router import AllProvidersFailed, Router
from app.schemas import (
    ChatChunk, ChatRequest, ChatResponse, Choice, ChoiceMessage, DeltaMessage,
    StreamChoice, Usage,
)
from app.semantic import (
    Embedder, GeminiEmbedder, HashingEmbedder, SemanticIndex, embed_text, partition_key,
)

log = logging.getLogger("gateway")
STATIC = Path(__file__).parent / "static"
HIT_HEADERS = {"hit": "HIT", "semantic": "SEMANTIC-HIT", "coalesced": "COALESCED"}


def _estimate_usage(request: ChatRequest, text: str) -> Usage:
    prompt = sum(len(m.content.split()) for m in request.messages)
    completion = len(text.split())
    return Usage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def _sse(chunk: ChatChunk) -> str:
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


def create_app(
    settings: Optional[Settings] = None,
    providers: Optional[list[Provider]] = None,
    redis_client=None,
    embedder: Optional[Embedder] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    providers = providers if providers is not None else build_providers(settings)

    owns_redis = False
    if redis_client is None and settings.redis_url:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(settings.redis_url)
        owns_redis = True

    if redis_client is not None:
        limiter = RedisTokenBucketLimiter(
            redis_client, settings.rate_limit_rpm, settings.rate_limit_burst
        )
        cache = RedisCache(redis_client, settings.cache_ttl_seconds)
    else:
        limiter = TokenBucketLimiter(settings.rate_limit_rpm, settings.rate_limit_burst)
        cache = MemoryCache(settings.cache_ttl_seconds, settings.cache_max_entries)

    semantic_index: Optional[SemanticIndex] = None
    if settings.semantic_cache_enabled:
        semantic_index = SemanticIndex(settings.semantic_max_entries)
        if embedder is None:
            if settings.semantic_embedder == "gemini" and settings.gemini_api_key:
                embedder = GeminiEmbedder(
                    settings.gemini_api_key, model=settings.semantic_embedding_model
                )
            else:
                embedder = HashingEmbedder()

    router = Router(
        providers,
        RetryPolicy(settings.retry_max_attempts, settings.retry_base_delay, settings.retry_max_delay),
        {
            p.name: CircuitBreaker(
                settings.breaker_failure_threshold, settings.breaker_recovery_seconds
            )
            for p in providers
        },
    )
    flight = SingleFlight()
    request_log = RequestLog(settings.database_url)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        for p in providers:
            await p.aclose()
        if embedder is not None and hasattr(embedder, "aclose"):
            await embedder.aclose()
        if owns_redis:
            await redis_client.aclose()
        request_log.close()

    app = FastAPI(
        title="LLM Gateway",
        version="1.0.0",
        description="OpenAI-compatible gateway: auth, rate limiting, caching, failover, streaming.",
        lifespan=lifespan,
    )

    def authenticate(authorization: Optional[str] = Header(default=None)) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        token = authorization.removeprefix("Bearer ").strip()
        if not any(hmac.compare_digest(token, k) for k in settings.api_keys):
            raise HTTPException(401, "Invalid API key")
        return hashlib.sha256(token.encode()).hexdigest()[:12]  # never log raw keys

    # ------------------------------------------------------------------ helpers
    def record(request: ChatRequest, key_id: str, start: float, **kw) -> None:
        request_log.record(
            key_id=key_id, model=kw.pop("model", request.model),
            latency_ms=(time.perf_counter() - start) * 1000, stream=request.stream, **kw,
        )

    async def semantic_lookup(request: ChatRequest):
        """Returns (cached_response, embedding, similarity). Fails open on embedder errors."""
        if semantic_index is None or embedder is None:
            return None, None, None
        try:
            vec = await embedder.embed(embed_text(request))
        except Exception as exc:
            log.warning("embedding failed, skipping semantic cache: %s", exc)
            return None, None, None
        hit = semantic_index.lookup(partition_key(request), vec, settings.semantic_threshold)
        if hit:
            hit_key, score = hit
            cached = await cache.get(hit_key)
            if cached is not None:
                return cached, vec, score
        return None, vec, None

    async def store(request: ChatRequest, key: Optional[str], vec, response: ChatResponse):
        if key is None:
            return
        await cache.set(key, response)
        if semantic_index is not None and vec is not None:
            semantic_index.add(partition_key(request), key, vec)

    async def replay_stream(request: ChatRequest, cached: ChatResponse) -> AsyncIterator[str]:
        """Stream a cached answer back in chunks so streaming clients see the same shape."""
        base = dict(id=cached.id, created=cached.created, model=cached.model)
        yield _sse(ChatChunk(**base, choices=[StreamChoice(delta=DeltaMessage(role="assistant", content=""))]))
        for piece in re.findall(r"\S+\s*", cached.choices[0].message.content):
            yield _sse(ChatChunk(**base, choices=[StreamChoice(delta=DeltaMessage(content=piece))]))
        yield _sse(ChatChunk(**base, choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")]))
        yield "data: [DONE]\n\n"

    # ---------------------------------------------------------------- endpoints
    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/dashboard", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC / "dashboard.html")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "providers": router.states(),
            "cache": cache.kind,
            "rate_limiter": limiter.kind,
            "semantic_cache": embedder.name if semantic_index is not None else None,
            "database": request_log.dialect,
        }

    @app.post(
        "/v1/chat/completions",
        response_model=None,
        responses={200: {"model": ChatResponse}},
    )
    async def chat_completions(request: ChatRequest, key_id: str = Depends(authenticate)):
        start = time.perf_counter()

        allowed, retry_after = await limiter.acquire(key_id)
        if not allowed:
            record(request, key_id, start, provider=None, cache_status="none", status=429)
            raise HTTPException(
                429, "Rate limit exceeded",
                headers={"Retry-After": str(max(1, int(retry_after + 0.999)))},
            )

        key = cache_key(request) if is_cacheable(request) else None
        cached: Optional[ChatResponse] = None
        vec, similarity = None, None
        cache_status = "miss"
        if key:
            cached = await cache.get(key)
            if cached is not None:
                cache_status = "hit"
            else:
                cached, vec, similarity = await semantic_lookup(request)
                if cached is not None:
                    cache_status = "semantic"

        # ---- served from cache ----
        if cached is not None:
            headers = {"X-Cache": HIT_HEADERS[cache_status]}
            if similarity is not None:
                headers["X-Cache-Similarity"] = f"{similarity:.4f}"
            record(request, key_id, start, provider="cache", cache_status=cache_status,
                   status=200, model=cached.model)
            if request.stream:
                return StreamingResponse(
                    replay_stream(request, cached), media_type="text/event-stream", headers=headers
                )
            return JSONResponse(cached.model_dump(), headers=headers)

        # ---- streaming from a provider ----
        if request.stream:
            try:
                sres = await router.stream(request)
            except AllProvidersFailed as exc:
                record(request, key_id, start, provider=None, cache_status="miss", status=502)
                raise HTTPException(502, f"All providers failed: {exc}")

            async def generate() -> AsyncIterator[str]:
                base = dict(
                    id=f"chatcmpl-{hashlib.sha1(str(time.time_ns()).encode()).hexdigest()[:24]}",
                    created=int(time.time()), model=request.model,
                )
                parts: list[str] = []
                usage: Optional[Usage] = None
                ttft: Optional[float] = None
                status, finished = 200, False
                try:
                    yield _sse(ChatChunk(**base, choices=[
                        StreamChoice(delta=DeltaMessage(role="assistant", content=""))]))
                    async for event in sres.events:
                        if event.delta:
                            if ttft is None:
                                ttft = (time.perf_counter() - start) * 1000
                            parts.append(event.delta)
                            yield _sse(ChatChunk(**base, choices=[
                                StreamChoice(delta=DeltaMessage(content=event.delta))]))
                        if event.usage:
                            usage = event.usage
                    yield _sse(ChatChunk(**base, choices=[
                        StreamChoice(delta=DeltaMessage(), finish_reason="stop")]))
                    yield "data: [DONE]\n\n"
                    finished = True
                except ProviderError as exc:
                    status = 502
                    yield f'data: {{"error": {{"message": "{type(exc).__name__}: stream interrupted"}}}}\n\n'
                    yield "data: [DONE]\n\n"
                except BaseException:
                    status = 499  # client disconnected mid-stream
                    raise
                finally:
                    text = "".join(parts)
                    final_usage = usage or _estimate_usage(request, text)
                    record(
                        request, key_id, start, provider=sres.provider, cache_status="miss",
                        status=status, ttft_ms=ttft, attempts=sres.attempts,
                        prompt_tokens=final_usage.prompt_tokens,
                        completion_tokens=final_usage.completion_tokens,
                        cost_usd=estimate_cost(
                            request.model, final_usage.prompt_tokens, final_usage.completion_tokens
                        ),
                    )
                if finished:
                    await store(request, key, vec, ChatResponse(
                        model=request.model,
                        choices=[Choice(message=ChoiceMessage(content=text))],
                        usage=final_usage,
                    ))

            return StreamingResponse(
                generate(), media_type="text/event-stream",
                headers={"X-Cache": "MISS", "X-Gateway-Provider": sres.provider,
                         "X-Gateway-Attempts": str(sres.attempts)},
            )

        # ---- plain request/response ----
        async def produce():
            routed = await router.chat(request)
            await store(request, key, vec, routed.response)
            return routed

        try:
            if key:
                routed, coalesced = await flight.run(key, produce)
            else:
                routed, coalesced = await produce(), False
        except AllProvidersFailed as exc:
            record(request, key_id, start, provider=None, cache_status="miss", status=502)
            raise HTTPException(502, f"All providers failed: {exc}")

        result = routed.response
        if coalesced:
            record(request, key_id, start, provider="coalesced", cache_status="coalesced",
                   status=200, model=result.model)
            return JSONResponse(result.model_dump(), headers={"X-Cache": "COALESCED"})

        record(
            request, key_id, start, provider=routed.provider, cache_status="miss", status=200,
            model=result.model, attempts=routed.attempts,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            cost_usd=estimate_cost(
                result.model, result.usage.prompt_tokens, result.usage.completion_tokens
            ),
        )
        return JSONResponse(
            result.model_dump(),
            headers={"X-Cache": "MISS", "X-Gateway-Provider": routed.provider,
                     "X-Gateway-Attempts": str(routed.attempts)},
        )

    @app.get("/v1/stats")
    def stats(
        window_minutes: Optional[int] = Query(default=None, ge=1, le=10080),
        _: str = Depends(authenticate),
    ) -> dict:
        return request_log.stats(window_minutes)

    @app.get("/v1/stats/timeseries")
    def timeseries(
        minutes: int = Query(default=30, ge=1, le=1440), _: str = Depends(authenticate)
    ) -> list[dict]:
        return request_log.timeseries(minutes)

    return app


app = create_app()
