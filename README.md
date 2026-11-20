# LLM Gateway

[![CI](https://github.com/charu698/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/charu698/llm-gateway/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

A production-style **OpenAI-compatible gateway** that sits in front of LLM providers and
handles what every LLM app otherwise re-implements: **auth, distributed rate limiting,
exact + semantic caching, retries with backoff, circuit breakers, provider failover,
streaming, and cost / latency observability.**

Point any OpenAI-style client at it by changing only the `base_url`.

![Dashboard](docs/dashboard.jpg)

## Headline results (measured, reproducible)

All numbers come from `benchmarks/run_benchmarks.sh` ([k6](https://k6.io)) with a simulated
200 ms upstream LLM, so they measure the *gateway*, not a vendor API. Setup and caveats are
[below](#benchmark-methodology-and-caveats).

| What | Result |
|---|---|
| Throughput, cache-heavy traffic (200 concurrent users, 1 process) | **6,322 req/s**, p95 **35 ms**, 0 errors |
| Latency of a cache hit (10 concurrent users) | p50 **1.6 ms**, p95 **2.3 ms**, vs **213 ms** p95 for an uncached request |
| Upstream calls avoided (189,741 requests, 100 distinct prompts) | only **100** reached the provider (**99.95%** served from cache / coalescing) |
| Success rate when the upstream fails 30% of calls | **70.0%** no protection → **97.5%** with retries → **100%** (0 failures / 19,451 requests) with retries + failover |
| Semantic cache (Gemini embeddings, 100 labelled prompt pairs) | **86%** of paraphrases served from cache at a **4%** false-hit rate |
| Gateway overhead on an uncached request (10 users, light load) | about **9 ms** p50 on top of the 200 ms upstream |
| Tests | **73** passing, including against real Redis and Postgres in CI |

## Features

- **OpenAI-compatible API**: `POST /v1/chat/completions`, including `stream: true` (Server-Sent Events).
- **Providers**: Gemini (native adapter) and any OpenAI-style endpoint (Groq, OpenRouter, vLLM, Ollama...) through one generic adapter. Ordered failover across them.
- **Resilience**: retries with exponential backoff + jitter for transient errors only (timeouts, 429, 5xx); per-provider **circuit breakers** (closed / open / half-open probe); stream failover before the first token.
- **Caching**: exact-match LRU/TTL cache (in-process or **Redis**), **semantic cache** (embedding similarity), and **request coalescing** so N identical in-flight requests cause one upstream call.
- **Rate limiting**: per-key token bucket, atomic **Redis Lua script** for multi-instance deployments (uses Redis's clock, fails open if Redis is down).
- **Observability**: every request logged (latency, status, provider, cache result, tokens, estimated cost, retries, time-to-first-token) to **SQLite or Postgres** via a batched background writer; `/v1/stats`, `/v1/stats/timeseries`, and a live **dashboard** at `/dashboard`.
- **Security**: API keys compared in constant time; raw keys never stored or logged (SHA-256 prefix only).
- **Ops**: Dockerfile, Docker Compose (gateway + Redis + Postgres), Render blueprint, GitHub Actions CI with Redis and Postgres service containers.

## Architecture

```
                    ┌──────────────────────────── LLM Gateway (FastAPI) ───────────────────────────┐
client ── POST ──►  │ 1 Auth ─► 2 Rate limit ─► 3 Cache ─► 4 Coalesce ─► 5 Router ─► Providers      │
/v1/chat/           │  Bearer    token bucket   exact ▸      N identical   retries +    ┌ Gemini     │
completions         │  (hashed   memory/Redis   semantic     requests →    backoff +    ├ Groq       │
 (stream: true      │   in logs) 429+Retry-     X-Cache:     1 upstream    breaker +    ├ OpenRouter │
  supported)        │            After          HIT/SEMANTIC call          failover     └ Mock/flaky │
                    │                                                                              │
                    │ 6 Batched async log writer ─► SQLite / Postgres ─► /v1/stats · /dashboard    │
                    └──────────────────────────────────────────────────────────────────────────────┘
```

Response headers expose what happened: `X-Cache` (`HIT`, `MISS`, `SEMANTIC-HIT`, `COALESCED`),
`X-Cache-Similarity`, `X-Gateway-Provider`, `X-Gateway-Attempts`.

## Quick start

```bash
git clone https://github.com/charu698/llm-gateway.git
cd llm-gateway

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # add GEMINI_API_KEY for real Gemini; empty = offline mock provider
uvicorn app.main:app --reload
```

- Interactive API docs: <http://localhost:8000/docs> (click **Authorize**, enter `dev-key-1`)
- Dashboard: <http://localhost:8000/dashboard>

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.5-flash","messages":[{"role":"user","content":"Hello"}]}'

# streaming
curl -N localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer dev-key-1" -H "Content-Type: application/json" \
  -d '{"model":"gemini-3.5-flash","stream":true,"messages":[{"role":"user","content":"Hello"}]}'
```

Using the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dev-key-1")
print(client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "Explain an API gateway in one line."}],
).choices[0].message.content)
```

### Full stack with Docker Compose (4 workers + Redis + Postgres)

```bash
docker compose up --build
```

### Deploy

`render.yaml` is a Render Blueprint that provisions the gateway plus managed Redis and Postgres.
Connect the repo in Render, then paste `GEMINI_API_KEY` in the dashboard (it is never committed).

## Semantic cache: what it does and how well it works

A new prompt whose embedding is within a cosine-similarity threshold of a cached prompt reuses that answer
(`X-Cache: SEMANTIC-HIT`). Entries are partitioned by model, `max_tokens` and system prompt, and only
deterministic requests (`temperature` 0 or unset) are cached.

I measured it on a hand-labelled set of 50 paraphrase pairs (should hit) and 50 look-alike-but-different
pairs such as "enable dark mode" vs "disable dark mode" (must not hit):
`python benchmarks/semantic_eval.py --gemini`.

| Embedder | Threshold | Paraphrases served from cache | Wrong answers served |
|---|---:|---:|---:|
| Gemini `gemini-embedding-2` | 0.90 | 94% | 14% |
| Gemini `gemini-embedding-2` | **0.92 (default)** | **86%** | **4%** |
| Gemini `gemini-embedding-2` | 0.94 | 68% | 4% |
| Gemini `gemini-embedding-2` | 0.96 | 30% | 2% |
| Offline hashing (default, no API) | any | ~0% | 2%+ |

Take-aways:
- The threshold is a precision / recall dial, so it is configurable (`SEMANTIC_THRESHOLD`).
- The built-in offline `hashing` embedder only catches near-identical wording; for real paraphrases
  use `SEMANTIC_EMBEDDER=gemini`.
- Semantic caching can return a wrong answer for a near-miss prompt (the 4% above). Enable it only
  where that trade-off is acceptable. It is **off by default**.
- The evaluation set is small and written by the project author, so treat the numbers as indicative.

## Reliability under failure

Upstream simulated to fail 30% of calls (503-style), 50 concurrent users, 30 s per scenario:

| Configuration | Success rate | p95 latency |
|---|---:|---:|
| Direct to the flaky provider (no retries) | 70.0% | 56 ms |
| + retries with exponential backoff (3 attempts) | 97.5% | 215 ms |
| + failover to a second provider and circuit breaker | **100%** (0 / 19,451 failed) | 216 ms |

Retries only fire for transient errors; permanent ones (bad model name, auth, safety block) skip
straight to the next provider. A circuit breaker stops hammering a provider that keeps failing and
probes it again after a cool-down. During development real Gemini models were retired
(404) and others returned intermittent `503 high demand`, which is exactly what this absorbs.

## Benchmark methodology and caveats

- Run with `benchmarks/run_benchmarks.sh` (needs k6). Raw JSON results and `SUMMARY.md` are in `benchmarks/results/`.
- Hardware: a fanless 8-core, 8 GB laptop running **everything** on one machine: the gateway,
  k6, Redis and Postgres. k6 itself uses a lot of CPU, so absolute throughput is conservative
  and noisy (multi-process runs varied between ~3,600 and ~5,500 req/s between repeats).
- The LLM is simulated (`MockProvider`, 200 ms), so numbers show gateway behaviour, not vendor latency.
- "Cache-heavy" = 100 distinct prompts, `temperature: 0`.

| scenario | VUs | req/s | success | p50 ms | p95 ms | p99 ms | upstream calls / requests |
|---|---:|---:|---:|---:|---:|---:|---:|
| uncached, light load | 10 | 48 | 100% | 209.0 | 213.3 | 225.2 | 1,441 / 1,441 |
| uncached, heavy load | 100 | 425 | 100% | 235.0 | 245.7 | 277.5 | 12,800 / 12,800 |
| cache hits, light load | 10 | 5,354 | 100% | 1.6 | 2.3 | 2.7 | 100 / 160,639 |
| cache-heavy, 1 process, in-memory | 200 | 6,322 | 100% | 30.5 | 35.1 | 53.0 | 100 / 189,741 |
| cache-heavy, 1 process, Redis | 200 | 4,221 | 100% | 44.2 | 66.3 | 83.5 | 148 / 126,721 |
| cache-heavy, 4 processes, Redis | 200 | 3,603 | 100% | 50.0 | 81.4 | 162.5 | 101 / 108,207 |
| cache-heavy, 4 processes, Redis + Postgres | 200 | 5,554 | 100% | 28.8 | 72.8 | 142.3 | 218 / 166,740 |

Honest reading of these: Redis adds a network round trip per request, so a single process is faster
with the in-process cache. Redis is what makes multi-process / multi-instance deployments consistent:
with 4 processes sharing it, 100 distinct prompts caused only 101 upstream calls (the cache and
rate limits are genuinely shared). I do not claim linear scaling from the 4-process runs on this laptop.

## Configuration

All settings are environment variables (or a local `.env`); see `.env.example` for the full list.

| Variable | Default | Description |
|---|---|---|
| `GATEWAY_API_KEYS` | `dev-key-1` | Comma-separated keys clients send as `Bearer` tokens |
| `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY` | empty | Enable that provider (missing key = skipped; none = mock only) |
| `PROVIDER_ORDER` | `gemini,groq,openrouter,mock` | Failover order |
| `RETRY_MAX_ATTEMPTS`, `RETRY_BASE_DELAY` | `3`, `0.2` | Retry policy per provider |
| `BREAKER_FAILURE_THRESHOLD`, `BREAKER_RECOVERY_SECONDS` | `5`, `30` | Circuit breaker |
| `RATE_LIMIT_RPM`, `RATE_LIMIT_BURST` | `60`, `10` | Token bucket per API key |
| `CACHE_TTL_SECONDS`, `CACHE_MAX_ENTRIES` | `300`, `1000` | Exact cache |
| `SEMANTIC_CACHE`, `SEMANTIC_EMBEDDER`, `SEMANTIC_THRESHOLD` | `false`, `hashing`, `0.92` | Semantic cache |
| `REDIS_URL` | empty | Shared cache + rate limiter (empty = in-process) |
| `DATABASE_URL` | `sqlite:///gateway.db` | SQLite or Postgres request log |

Cost estimates use the `PRICING` table in `app/db.py` (USD per 1M tokens); models not listed are costed at 0.

## Project structure

```
app/
  main.py          App factory, endpoints, request pipeline (auth -> limit -> cache -> route)
  router.py        Retries, circuit breakers, ordered failover (chat + streaming)
  resilience.py    RetryPolicy (backoff + jitter) and CircuitBreaker
  cache.py         Memory / Redis cache, request coalescing (SingleFlight)
  semantic.py      Embedders (offline + Gemini) and the vector index
  ratelimit.py     Token bucket: in-process and Redis (Lua)
  db.py            SQLAlchemy request log, batched writer, stats, time series
  providers/       Provider interface; Gemini, OpenAI-compatible, mock adapters
  static/          Dashboard
benchmarks/        k6 load test, runner, semantic-cache evaluation, results
tests/             73 tests (unit + integration; Redis/Postgres suites run when available)
```

## Tests

```bash
pytest -q                                   # 66 pass, 7 skip (Redis/Postgres suites skipped)

REDIS_URL=redis://localhost:6379/0 \
TEST_DATABASE_URL=postgresql://user:pass@localhost:5432/test pytest -q   # all 73
```

Covers: auth, OpenAI response shape, exact/semantic/coalesced caching, retries with backoff,
circuit-breaker state machine, failover (including before the first streamed token), streaming
format and caching of streamed answers, provider adapters (against `httpx.MockTransport`),
Redis limiter atomicity under concurrency, fail-open behaviour when Redis is down, two gateway
instances sharing cache and rate limit, batched log writes, and SQLite + Postgres stats.

## Known limitations

- Verified live against Gemini (non-streaming, streaming, embeddings). The Groq/OpenRouter adapter
  is covered by mock-transport tests but has not been run against those APIs.
- `docker-compose.yml`, the Dockerfile and `render.yaml` are provided but not run on the author's machine
  (no Docker locally); CI exercises Redis and Postgres via service containers instead.
- Request coalescing and the semantic vector index are per-process (cached answers are shared via Redis).
- Stats percentiles are computed over the most recent 100k successful requests.
- Cost estimates need prices filled in `PRICING`.

## License

MIT
