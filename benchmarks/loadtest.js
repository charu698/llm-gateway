// k6 load test for the gateway.
//
//   k6 run -e NAME=cache_heavy -e PROMPTS=100 -e VUS=200 -e DURATION=30s benchmarks/loadtest.js
//
// Env:  URL (default http://127.0.0.1:8800)  KEY (default dev-key-1)
//       PROMPTS  number of distinct prompts; 0 = every request unique (always a cache miss)
//       VUS, DURATION, NAME (results are written to benchmarks/results/<NAME>.json)
import http from 'k6/http';
import { check } from 'k6';
import { Counter, Rate } from 'k6/metrics';

const URL = __ENV.URL || 'http://127.0.0.1:8800';
const KEY = __ENV.KEY || 'dev-key-1';
const PROMPTS = parseInt(__ENV.PROMPTS || '0', 10);
const NAME = __ENV.NAME || 'run';

const success = new Rate('gw_success');           // HTTP 200
const cacheServed = new Counter('gw_cache_served'); // HIT / SEMANTIC-HIT / COALESCED
const upstream = new Counter('gw_upstream');        // MISS (went to a provider)
const limited = new Counter('gw_rate_limited');     // HTTP 429

export const options = {
  scenarios: {
    load: { executor: 'constant-vus', vus: parseInt(__ENV.VUS || '100', 10), duration: __ENV.DURATION || '30s' },
  },
  summaryTrendStats: ['avg', 'med', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  // One random index, so there are exactly PROMPTS distinct prompts.
  const text = PROMPTS > 0
    ? `Question number ${Math.floor(Math.random() * PROMPTS)}: explain the topic`
    : `unique ${__VU}-${__ITER}-${Date.now()}`;
  const res = http.post(
    `${URL}/v1/chat/completions`,
    JSON.stringify({ model: 'bench-model', temperature: 0, messages: [{ role: 'user', content: text }] }),
    { headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` } },
  );
  const ok = res.status === 200;
  success.add(ok);
  if (res.status === 429) limited.add(1);
  const cache = res.headers['X-Cache'];
  if (ok && cache === 'MISS') upstream.add(1);
  else if (ok && cache) cacheServed.add(1);
  check(res, { 'status is 200': () => ok });
}

export function handleSummary(data) {
  const m = data.metrics;
  const d = m.http_req_duration.values;
  const out = {
    name: NAME,
    requests: m.http_reqs.values.count,
    rps: m.http_reqs.values.rate,
    success_rate: m.gw_success ? m.gw_success.values.rate : null,
    cache_served: m.gw_cache_served ? m.gw_cache_served.values.count : 0,
    upstream_calls: m.gw_upstream ? m.gw_upstream.values.count : 0,
    rate_limited: m.gw_rate_limited ? m.gw_rate_limited.values.count : 0,
    latency_ms: { avg: d.avg, p50: d.med, p90: d['p(90)'], p95: d['p(95)'], p99: d['p(99)'], max: d.max },
    vus: parseInt(__ENV.VUS || '100', 10),
    duration: __ENV.DURATION || '30s',
    distinct_prompts: PROMPTS,
  };
  const line = `${NAME}: ${out.requests} req, ${out.rps.toFixed(0)} req/s, success ${(out.success_rate * 100).toFixed(2)}%, ` +
    `p50 ${d.med.toFixed(1)} ms, p95 ${d['p(95)'].toFixed(1)} ms, p99 ${d['p(99)'].toFixed(1)} ms\n`;
  return { [`benchmarks/results/${NAME}.json`]: JSON.stringify(out, null, 2), stdout: line };
}
