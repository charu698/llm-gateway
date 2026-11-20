# LLM Gateway

An OpenAI-compatible gateway that sits in front of LLM providers and adds the
production concerns you don't want in every app: **auth, rate limiting,
response caching, provider failover, and cost/latency tracking**.

```
client ──► /v1/chat/completions
             │  1. API-key auth (keys hashed in logs)
             │  2. Token-bucket rate limit (per key)  → 429 + Retry-After
             │  3. Cache lookup (deterministic requests only) → X-Cache: HIT
             │  4. Provider router with fallback: Gemini → ... → mock
             │  5. Log latency, tokens, estimated cost to SQLite
             ▼
         /v1/stats  (hit rate, p95 latency, cost, error rate, per-provider split)
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set GEMINI_API_KEY to use real Gemini; empty = mock provider
uvicorn app.main:app --reload
```

Then open http://localhost:8000/docs for the interactive Swagger UI
(click **Authorize**, enter `dev-key-1`, and try `/v1/chat/completions`).

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello"}]}'

curl -s localhost:8000/v1/stats -H "Authorization: Bearer dev-key-1"
```

Because the response is OpenAI-shaped, existing OpenAI SDK clients work by
changing only `base_url`.

## Design notes

| Concern | Approach |
|---|---|
| Rate limiting | Token bucket per key (burst + steady refill), injectable clock for deterministic tests |
| Caching | Exact-match LRU + TTL, only for `temperature` unset/0 so non-deterministic calls are never replayed |
| Failover | Ordered provider list; `ProviderError` triggers fallback; all failing returns 502 |
| Observability | Every request logged (latency, tokens, cost, status, provider, cache hit) |
| Security | Raw API keys are never stored; only a SHA-256 prefix is logged |

## Tests

```bash
pytest -q
```

Covers auth, OpenAI response shape, caching, fallback, 502 on total failure,
rate limiting, stats, and token-bucket refill.

## Roadmap

- [ ] Streaming (SSE) responses
- [ ] Semantic cache (embeddings + similarity threshold)
- [ ] Redis-backed rate limiter and cache (multi-instance)
- [ ] Postgres request log
- [ ] More providers (Groq, OpenRouter) + retries with backoff / circuit breaker
- [ ] Dashboard (React) and k6 load-test benchmarks
- [ ] Docker Compose + live deployment
