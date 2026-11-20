"""Semantic cache: serve a cached answer to a *paraphrase* of an earlier prompt.

Prompts are embedded as unit vectors; a new prompt whose cosine similarity to a
cached prompt is above a threshold reuses that answer. The vector index is
per-process; the cached responses themselves live in the shared cache backend.
"""
import hashlib
import re
import zlib
from typing import Optional, Protocol

import httpx
import numpy as np

from app.schemas import ChatRequest

_WORD = re.compile(r"[a-z0-9]+")

# Function words carry little meaning. Negations ("not", "no", "never") and
# question words are deliberately NOT here: dropping them would make opposite
# questions look identical.
STOPWORDS = frozenset(
    "a an the is are was were be been being am do does did of to in on at for with by "
    "from as and or it its this that these those i you we they he she me my your our "
    "their s please can could would will shall just".split()
)


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class Embedder(Protocol):
    name: str

    async def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalised float32 vector."""


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


class HashingEmbedder:
    """Dependency-free, offline embedder: hashed unigram + bigram features.

    Cheap (microseconds) and deterministic, but lexical: it catches rewordings
    that share vocabulary, not synonyms. `GeminiEmbedder` handles those.
    """

    name = "hashing"

    def __init__(self, dims: int = 2048):
        self.dims = dims

    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode()) % self.dims

    def embed_sync(self, text: str) -> np.ndarray:
        tokens = [_stem(w) for w in _WORD.findall(text.lower())]
        vec = np.zeros(self.dims, dtype=np.float32)
        for tok in tokens:
            if tok not in STOPWORDS:
                vec[self._bucket(tok)] += 1.0
        for a, b in zip(tokens, tokens[1:]):
            if a in STOPWORDS and b in STOPWORDS:
                continue
            vec[self._bucket(f"{a} {b}")] += 0.6
        return _normalise(vec)

    async def embed(self, text: str) -> np.ndarray:
        return self.embed_sync(text)


class GeminiEmbedder:
    """Dense semantic embeddings from the Gemini embedding API."""

    name = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        dims: int = 768,
        timeout: float = 15.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.dims = dims
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def embed(self, text: str) -> np.ndarray:
        resp = await self._client.post(
            f"{self.BASE_URL}/{self.model}:embedContent",
            headers={"x-goog-api-key": self.api_key},
            json={
                "content": {"parts": [{"text": text}]},
                "taskType": "SEMANTIC_SIMILARITY",
                "outputDimensionality": self.dims,
            },
        )
        resp.raise_for_status()
        return _normalise(np.asarray(resp.json()["embedding"]["values"], dtype=np.float32))

    async def aclose(self) -> None:
        await self._client.aclose()


def embed_text(request: ChatRequest) -> str:
    return "\n".join(m.content for m in request.messages if m.role != "system")


def partition_key(request: ChatRequest) -> str:
    """Only compare prompts that share model, output limit and system prompt."""
    system = "\n".join(m.content for m in request.messages if m.role == "system")
    digest = hashlib.sha1(system.encode()).hexdigest()[:8]
    return f"{request.model}|{request.max_tokens}|{digest}"


class _Partition:
    def __init__(self, dims: int, capacity: int):
        self.capacity = capacity
        self.mat = np.zeros((min(capacity, 64), dims), dtype=np.float32)
        self.keys: list[Optional[str]] = []
        self.size = 0
        self.pos = 0  # next slot to overwrite once full (ring buffer)
        self.seen: set[str] = set()

    def add(self, key: str, vec: np.ndarray) -> None:
        if key in self.seen:
            return
        if self.size < self.capacity:
            if self.size == self.mat.shape[0]:  # grow geometrically
                grown = np.zeros((min(self.capacity, self.size * 2), self.mat.shape[1]), np.float32)
                grown[: self.size] = self.mat[: self.size]
                self.mat = grown
            slot = self.size
            self.size += 1
            self.keys.append(key)
        else:
            slot = self.pos
            self.seen.discard(self.keys[slot])
            self.keys[slot] = key
            self.pos = (self.pos + 1) % self.capacity
        self.mat[slot] = vec
        self.seen.add(key)


class SemanticIndex:
    def __init__(self, max_entries: int = 5000):
        self.max_entries = max_entries
        self._parts: dict[str, _Partition] = {}

    def __len__(self) -> int:
        return sum(p.size for p in self._parts.values())

    def add(self, partition: str, key: str, vec: np.ndarray) -> None:
        part = self._parts.get(partition)
        if part is None:
            part = self._parts[partition] = _Partition(vec.shape[0], self.max_entries)
        part.add(key, vec)

    def lookup(
        self, partition: str, vec: np.ndarray, threshold: float
    ) -> Optional[tuple[str, float]]:
        part = self._parts.get(partition)
        if part is None or part.size == 0 or part.mat.shape[1] != vec.shape[0]:
            return None
        scores = part.mat[: part.size] @ vec  # unit vectors, so dot product = cosine
        best = int(np.argmax(scores))
        score = float(scores[best])
        if score >= threshold:
            return part.keys[best], score  # type: ignore[return-value]
        return None
