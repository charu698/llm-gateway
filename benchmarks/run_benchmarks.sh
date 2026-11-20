#!/usr/bin/env bash
# Reproducible benchmark suite. Starts the gateway in several configurations, drives each
# with k6, and writes JSON results to benchmarks/results/ (then SUMMARY.md).
#
#   REDIS_URL=redis://localhost:6379/0 DATABASE_URL=postgresql://... benchmarks/run_benchmarks.sh
#
# Needs: k6, python deps from requirements.txt, and (optionally) a running Redis / Postgres.
# The upstream LLM is simulated (MockProvider) so results measure the *gateway*, not an API.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
PORT=${PORT:-8800}
DURATION=${DURATION:-30s}
REDIS_REAL=${REDIS_URL:-}   # remembered; REDIS_URL itself is toggled per scenario
PG_URL=${DATABASE_URL:-}
mkdir -p benchmarks/results
PID=""

start_gateway() { # args: workers  (env for the gateway is taken from the caller's exported vars)
  local workers=$1
  $PY -m uvicorn app.main:app --port "$PORT" --workers "$workers" --log-level warning \
      > benchmarks/results/gateway.log 2>&1 &
  PID=$!
  for _ in $(seq 1 60); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && return 0; sleep 0.5; done
  echo "gateway failed to start"; cat benchmarks/results/gateway.log; exit 1
}
stop_gateway() { [ -n "$PID" ] && kill "$PID" 2>/dev/null && wait "$PID" 2>/dev/null || true; PID=""; }
trap stop_gateway EXIT

# Settings shared by every run: no rate limiting in the way, simulated 200 ms LLM.
export GATEWAY_API_KEYS=dev-key-1 RATE_LIMIT_RPM=100000000 RATE_LIMIT_BURST=100000000
export PROVIDER_ORDER=mock MOCK_LATENCY_MS=200 CACHE_TTL_SECONDS=3600 CACHE_MAX_ENTRIES=100000

run() { # name workers prompts vus duration [extra k6 env...]
  local name=$1 workers=$2 prompts=$3 vus=$4 dur=$5
  echo "=== $name (workers=$workers prompts=$prompts vus=$vus)"
  rm -f benchmarks/results/bench.db*
  [ -n "${REDIS_FLUSH:-}" ] && redis-cli -u "$REDIS_REAL" flushdb >/dev/null || true
  start_gateway "$workers"
  k6 run --quiet -e NAME="$name" -e PROMPTS="$prompts" -e VUS="$vus" -e DURATION="$dur" \
      -e URL="http://127.0.0.1:$PORT" benchmarks/loadtest.js 2>&1 | grep -E "^$name" || true
  stop_gateway
}

# ---- 1. Gateway overhead: every request is a cache miss to a 200 ms upstream --------------
export DATABASE_URL=sqlite:///benchmarks/results/bench.db REDIS_URL=""
run overhead_uncached_10vu 1 0 10 "$DURATION"      # light load: overhead without queueing
run overhead_uncached_100vu 1 0 100 "$DURATION"    # heavy load
run cache_hit_latency_10vu 1 100 10 "$DURATION"    # light load: latency of a cache hit

# ---- 2. Cache-heavy traffic (100 distinct prompts): single worker, in-process cache ------
run cache_heavy_1w_memory 1 100 200 "$DURATION"

if [ -n "$REDIS_REAL" ]; then
  export REDIS_FLUSH=1
  # ---- 3. Same workload with Redis (shared cache + distributed rate limiter) --------------
  export REDIS_URL="$REDIS_REAL"
  run cache_heavy_1w_redis 1 100 200 "$DURATION"
  # ---- 4. Scale out: 4 worker processes sharing one Redis ---------------------------------
  run cache_heavy_4w_redis 4 100 200 "$DURATION"
  # ---- 5. Full stack: 4 workers + Redis + Postgres request log -----------------------------
  if [ -n "$PG_URL" ]; then
    export DATABASE_URL=$PG_URL
    run cache_heavy_4w_redis_postgres 4 100 200 "$DURATION"
    export DATABASE_URL=sqlite:///benchmarks/results/bench.db
  fi
  export REDIS_URL=""
fi

# ---- 6. Reliability: upstream that fails 30% of calls (HTTP 503-style) ----------------------
unset REDIS_FLUSH
export MOCK_LATENCY_MS=50 FLAKY_FAIL_RATE=0.3 REDIS_URL="" BREAKER_FAILURE_THRESHOLD=1000000
export PROVIDER_ORDER=flaky RETRY_MAX_ATTEMPTS=1
run flaky30_no_retries 1 0 50 "$DURATION"
export RETRY_MAX_ATTEMPTS=3 RETRY_BASE_DELAY=0.05
run flaky30_with_retries 1 0 50 "$DURATION"
export PROVIDER_ORDER=flaky,mock BREAKER_FAILURE_THRESHOLD=5 BREAKER_RECOVERY_SECONDS=5
run flaky30_retries_and_failover 1 0 50 "$DURATION"

$PY benchmarks/summarize.py
