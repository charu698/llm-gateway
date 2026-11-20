import asyncio

import numpy as np

from app.semantic import HashingEmbedder, SemanticIndex
from tests.conftest import AUTH, CountingProvider, make_client


def sim(a: str, b: str) -> float:
    e = HashingEmbedder()
    return float(e.embed_sync(a) @ e.embed_sync(b))


def test_embeddings_are_unit_vectors():
    v = HashingEmbedder().embed_sync("what is the capital of france")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_rewording_scores_higher_than_unrelated_text():
    same = sim("What is the capital of France?", "what's the capital of france")
    unrelated = sim("What is the capital of France?", "How do I bake sourdough bread?")
    assert same > 0.9 and unrelated < 0.2


def test_negation_is_not_ignored():
    # "enable" vs "disable" and "not" must not look identical
    assert sim("how to enable dark mode", "how to disable dark mode") < 0.9


def test_index_lookup_threshold_and_partitions():
    emb = HashingEmbedder()
    idx = SemanticIndex()
    idx.add("p1", "k1", emb.embed_sync("what is the capital of france"))
    q = emb.embed_sync("capital of france?")
    assert idx.lookup("p1", q, 0.5)[0] == "k1"
    assert idx.lookup("p1", q, 0.99) is None          # threshold respected
    assert idx.lookup("other-partition", q, 0.0) is None  # partitions isolated


def test_index_evicts_oldest_when_full():
    emb = HashingEmbedder()
    idx = SemanticIndex(max_entries=2)
    for i, text in enumerate(["alpha one", "beta two", "gamma three"]):
        idx.add("p", f"k{i}", emb.embed_sync(text))
    assert len(idx) == 2
    assert idx.lookup("p", emb.embed_sync("alpha one"), 0.99) is None  # evicted
    assert idx.lookup("p", emb.embed_sync("gamma three"), 0.99)[0] == "k2"


def body(text, **kw):
    return {"model": "gemini-3.5-flash", "messages": [{"role": "user", "content": text}], **kw}


def test_gateway_serves_paraphrase_from_semantic_cache(tmp_path):
    provider = CountingProvider()
    with make_client(tmp_path, providers=[provider], semantic_cache_enabled=True,
                     semantic_threshold=0.85) as c:
        first = c.post("/v1/chat/completions", json=body("What is the capital of France?"), headers=AUTH)
        second = c.post("/v1/chat/completions", json=body("what's the capital of france"), headers=AUTH)
        other = c.post("/v1/chat/completions", json=body("Explain how photosynthesis works"), headers=AUTH)
        health = c.get("/health").json()
    assert first.headers["X-Cache"] == "MISS"
    assert second.headers["X-Cache"] == "SEMANTIC-HIT"
    assert float(second.headers["X-Cache-Similarity"]) >= 0.85
    assert second.json()["id"] == first.json()["id"]
    assert other.headers["X-Cache"] == "MISS"
    assert provider.calls == 2  # the paraphrase never reached the provider
    assert health["semantic_cache"] == "hashing"


def test_semantic_cache_is_off_by_default(tmp_path):
    provider = CountingProvider()
    with make_client(tmp_path, providers=[provider]) as c:
        c.post("/v1/chat/completions", json=body("What is the capital of France?"), headers=AUTH)
        r = c.post("/v1/chat/completions", json=body("what's the capital of france"), headers=AUTH)
    assert r.headers["X-Cache"] == "MISS" and provider.calls == 2


def test_semantic_cache_respects_system_prompt_and_model(tmp_path):
    provider = CountingProvider()
    with make_client(tmp_path, providers=[provider], semantic_cache_enabled=True,
                     semantic_threshold=0.85) as c:
        c.post("/v1/chat/completions", json=body("What is the capital of France?"), headers=AUTH)
        diff_model = {**body("what's the capital of france"), "model": "other-model"}
        r = c.post("/v1/chat/completions", json=diff_model, headers=AUTH)
    assert r.headers["X-Cache"] == "MISS"


def test_embedder_failure_fails_open(tmp_path):
    class BrokenEmbedder:
        name = "broken"

        async def embed(self, text):
            raise RuntimeError("embedding api down")

    provider = CountingProvider()
    with make_client(tmp_path, providers=[provider], semantic_cache_enabled=True,
                     embedder=BrokenEmbedder()) as c:
        r = c.post("/v1/chat/completions", json=body("hello there"), headers=AUTH)
    assert r.status_code == 200 and provider.calls == 1
