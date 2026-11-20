"""Measure semantic-cache accuracy on a labelled set of prompt pairs.

For each embedder and cosine-similarity threshold:
  recall    = paraphrase pairs correctly served from cache  (cache hits you wanted)
  false-hit = different-meaning pairs wrongly served from cache  (wrong answers)

Usage:
  python benchmarks/semantic_eval.py                  # offline hashing embedder only
  GEMINI_API_KEY=... python benchmarks/semantic_eval.py --gemini
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.semantic import GeminiEmbedder, HashingEmbedder  # noqa: E402

THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96]


CACHE_DIR = ROOT / "benchmarks" / "results" / ".embedding_cache"


async def embed_all(embedder, texts: list[str], pace: float = 0.0) -> dict[str, np.ndarray]:
    """Embed every text, resuming from an on-disk cache and pacing API calls."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{embedder.name}_{getattr(embedder, 'model', 'local')}.json"
    cache: dict[str, list[float]] = json.loads(cache_file.read_text()) if cache_file.exists() else {}

    for i, text in enumerate(texts):
        if text in cache:
            continue
        for attempt in range(8):
            try:
                cache[text] = [float(x) for x in await embedder.embed(text)]
                break
            except Exception as exc:
                wait = min(60, 5 * (attempt + 1))
                print(f"  embed retry {attempt + 1} ({type(exc).__name__}: {str(exc)[:80]}), waiting {wait}s")
                await asyncio.sleep(wait)
        else:
            cache_file.write_text(json.dumps(cache))
            raise RuntimeError(f"embedding failed for: {text!r}")
        if pace:
            await asyncio.sleep(pace)
        if i % 20 == 0:
            cache_file.write_text(json.dumps(cache))
    cache_file.write_text(json.dumps(cache))
    return {t: np.asarray(cache[t], dtype=np.float32) for t in texts}


def evaluate(vectors, pairs_pos, pairs_neg):
    pos = np.array([float(vectors[a] @ vectors[b]) for a, b in pairs_pos])
    neg = np.array([float(vectors[a] @ vectors[b]) for a, b in pairs_neg])
    rows = []
    for t in THRESHOLDS:
        tp, fp = int((pos >= t).sum()), int((neg >= t).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        rows.append({
            "threshold": t,
            "recall": round(tp / len(pos), 3),
            "false_hit_rate": round(fp / len(neg), 3),
            "precision": round(precision, 3),
            "true_hits": tp, "false_hits": fp,
        })
    return rows, pos, neg


def best_recall_at_fpr(pos, neg, max_fpr: float):
    """Highest recall achievable while keeping false hits at or below max_fpr."""
    best = (0.0, None)
    for t in np.unique(np.concatenate([pos, neg])):
        if (neg >= t).mean() <= max_fpr and (pos >= t).mean() > best[0]:
            best = (float((pos >= t).mean()), float(t))
    return best


def print_table(name, rows):
    print(f"\n{name}")
    print(f"{'threshold':>9}  {'recall':>7}  {'false-hit':>9}  {'precision':>9}")
    for r in rows:
        print(f"{r['threshold']:>9.2f}  {r['recall']:>7.1%}  {r['false_hit_rate']:>9.1%}  {r['precision']:>9.1%}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemini", action="store_true")
    ap.add_argument("--model", default="gemini-embedding-2")
    args = ap.parse_args()

    data = json.loads((ROOT / "benchmarks" / "semantic_pairs.json").read_text())
    pos_pairs, neg_pairs = data["paraphrases"], data["different"]
    texts = sorted({t for pair in pos_pairs + neg_pairs for t in pair})
    print(f"{len(pos_pairs)} paraphrase pairs, {len(neg_pairs)} different-meaning pairs, "
          f"{len(texts)} unique prompts")

    embedders = [HashingEmbedder()]
    if args.gemini:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            for line in (ROOT / ".env").read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    key = line.split("=", 1)[1].strip()
        embedders.append(GeminiEmbedder(key, model=args.model))

    results = {}
    for emb in embedders:
        vectors = await embed_all(emb, texts, pace=0.8 if emb.name == "gemini" else 0.0)
        rows, pos, neg = evaluate(vectors, pos_pairs, neg_pairs)
        print_table(f"embedder = {emb.name}", rows)
        for fpr in (0.0, 0.05, 0.10):
            rec, thr = best_recall_at_fpr(pos, neg, fpr)
            print(f"  best recall at <= {fpr:.0%} false hits: {rec:.1%} (threshold {thr:.3f})"
                  if thr is not None else f"  no threshold reaches <= {fpr:.0%} false hits")
        results[emb.name] = {
            "sweep": rows,
            "best_recall_at_fpr": {
                f"{int(f * 100)}%": dict(zip(("recall", "threshold"), best_recall_at_fpr(pos, neg, f)))
                for f in (0.0, 0.05, 0.10)
            },
            "mean_similarity_paraphrase": round(float(pos.mean()), 3),
            "mean_similarity_different": round(float(neg.mean()), 3),
        }
        if hasattr(emb, "aclose"):
            await emb.aclose()

    out_dir = ROOT / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "semantic_eval.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_dir / 'semantic_eval.json'}")


if __name__ == "__main__":
    asyncio.run(main())
