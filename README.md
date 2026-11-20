# LLM Gateway

[![CI](https://github.com/charu698/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/charu698/llm-gateway/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688)
![License](https://img.shields.io/badge/license-MIT-green)

An **OpenAI-compatible gateway** that sits in front of LLM providers and adds the
production concerns you don't want to re-implement in every app:
**authentication, rate limiting, response caching, provider failover, and
cost / latency tracking.**

Point any OpenAI-style client at the gateway by changing only the `base_url`.

## Why this exists

Calling an LLM API directly from an app means every service has to handle keys,
quotas, outages, repeated identical prompts, and cost visibility on its own.
During development of this project, Gemini models were retired and others
returned intermittent `503 high demand` errors, which is exactly the failure
mode a gateway is meant to absorb: when the primary provider fails, the request
falls through to the next provider instead of failing the user.

## Architecture

```
                     ┌──────────────────────── LLM Gateway (FastAPI) ────────────────────────┐
                     │                                                                       │
client ── POST ──►   │  1. Auth ──► 2. Rate limit ──► 3. Cache ──► 4. Router ──► Providers   │
/v1/chat/completions │   Bearer      token bucket      exact-match    ordered     ┌ Gemini   │
                     │   key         per key           LRU + TTL      fallback    └ Mock/... │
                     │   (hashed      429 +            X-Cache:                              │
                     │    in logs)    Retry-After      HIT / MISS                            │
                     │                                                                       │
                     │  5. Log every request ──► SQLite ──► GET /v1/stats                    │
                     └───────────────────────────────────────────────────────────────────────┘
```

| Concern | Approach |
|---|---|
| **Auth** | Bearer API keys from config. Raw keys are never stored; only a SHA-256 prefix is logged. |
| **Rate limiting** | Per-key token bucket (burst + steady refill). Returns `429` with a `Retry-After` header. Clock is injectable, so refill behavior is tested deterministically. |
| **Caching** | Exact-match LRU with TTL, applied only when `temperature` is unset or `0`, so non-deterministic generations are never replayed. Responses carry `X-Cache: HIT/MISS`. |
| **Failover** | Providers are tried in configured order. A `ProviderError` triggers the next provider; if all fail the gateway returns `502`. The serving provider is exposed in `X-Gateway-Provider`. |
| **Observability** | Every request logs latency, status, provider, cache hit, token counts and estimated cost. `/v1/stats` reports hit rate, avg/p95 latency, error rate, tokens, cost and per-provider split. |
| **Provider abstraction** | A small `Provider` interface (`chat(request) -> response`). Adding a provider means writing one adapter class. |

## Quick start

```bash
git clone https://github.com/charu698/llm-gateway.git
cd llm-gateway

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # add GEMINI_API_KEY for real Gemini; leave empty to use the mock provider
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> for the interactive Swagger UI
(click **Authorize**, enter `dev-key-1`, then try `POST /v1/chat/completions`).

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello"}]}'

curl -s localhost:8000/v1/stats -H "Authorization: Bearer dev-key-1"
```

Example `/v1/stats` response:

```json
{
  "total_requests": 22,
  "cache_hit_rate": 0.0909,
  "error_rate": 0.2727,
  "avg_latency_ms": 1819.68,
  "p95_latency_ms": 18672.14,
  "total_tokens": 178,
  "total_cost_usd": 0.00003,
  "by_provider": { "gemini": 2, "mock": 12 }
}
```

### Using it from the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dev-key-1")
reply = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "Explain an API gateway in one line."}],
)
print(reply.choices[0].message.content)
```

## Configuration

All settings are environment variables (or a local `.env`):

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_API_KEYS` | `dev-key-1` | Comma-separated keys clients must send as `Bearer` tokens |
| `GEMINI_API_KEY` | _empty_ | Enables the Gemini provider; empty means mock only |
| `PROVIDER_ORDER` | `gemini,mock` | Fallback order |
| `RATE_LIMIT_RPM` | `60` | Steady-state requests per minute, per key |
| `RATE_LIMIT_BURST` | `10` | Bucket capacity per key |
| `CACHE_TTL_SECONDS` | `300` | Cache entry lifetime |
| `CACHE_MAX_ENTRIES` | `1000` | LRU capacity |
| `DB_PATH` | `gateway.db` | SQLite request log |

Cost tracking uses the `PRICING` table in `app/db.py` (USD per 1M tokens).
Models not listed are costed at 0; fill in current prices from your provider.

## Project structure

```
app/
  main.py          FastAPI app factory, endpoints, request pipeline
  config.py        Settings from environment / .env
  schemas.py       OpenAI-compatible request/response models
  ratelimit.py     Token-bucket limiter
  cache.py         LRU + TTL response cache
  router.py        Ordered provider fallback
  db.py            SQLite request log, stats, cost estimation
  providers/       Provider interface + Gemini and mock adapters
tests/             pytest suite
Dockerfile         Container image
.github/workflows  CI (runs pytest on every push)
```

## Tests

```bash
pytest -q
```

10 tests cover: auth rejection, OpenAI response shape, cache hit/miss,
non-deterministic requests bypassing the cache, provider fallback, `502` when
every provider fails, `429` + `Retry-After`, stats aggregation, and
token-bucket refill over time.

## Docker

```bash
docker build -t llm-gateway .
docker run -p 8000:8000 --env-file .env llm-gateway
```

## Known limitations

- Cache, rate limiter and request log are single-process (in-memory / SQLite),
  so running several instances would not share state yet.
- No streaming responses yet; replies are returned in one piece.
- Failover triggers on provider errors but does not yet retry with backoff.

## Roadmap

- [ ] Retries with exponential backoff and a circuit breaker
- [ ] Streaming (SSE) responses
- [ ] Additional providers (Groq, OpenRouter)
- [ ] Redis-backed rate limiter and cache for multi-instance deployments
- [ ] Postgres request log
- [ ] Semantic cache (embeddings + similarity threshold)
- [ ] k6 load tests with published benchmark numbers
- [ ] React dashboard, Docker Compose, live deployment

## License

MIT
