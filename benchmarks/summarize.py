"""Turn benchmarks/results/*.json into a Markdown table (benchmarks/results/SUMMARY.md)."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ORDER = [
    "overhead_uncached_10vu", "overhead_uncached_100vu", "cache_hit_latency_10vu",
    "cache_heavy_1w_memory", "cache_heavy_1w_redis", "cache_heavy_4w_redis",
    "cache_heavy_4w_redis_postgres",
    "flaky30_no_retries", "flaky30_with_retries", "flaky30_retries_and_failover",
]


def main() -> None:
    rows = []
    for name in ORDER:
        f = RESULTS / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        lat = r["latency_ms"]
        rows.append(
            f"| `{name}` | {r['vus']} | {r['rps']:,.0f} | {r['success_rate'] * 100:.2f}% | "
            f"{lat['p50']:.1f} | {lat['p95']:.1f} | {lat['p99']:.1f} | "
            f"{r['upstream_calls']:,} / {r['requests']:,} |"
        )
    table = (
        "| scenario | VUs | req/s | success | p50 ms | p95 ms | p99 ms | upstream calls / requests |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n"
    )
    (RESULTS / "SUMMARY.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
