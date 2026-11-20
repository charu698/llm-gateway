import asyncio
import json

import httpx
import numpy as np
import pytest

from app.semantic import GeminiEmbedder


def make(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiEmbedder("secret-key", client=client, **kw)


def test_gemini_embedder_request_and_normalisation():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"embedding": {"values": [3.0, 4.0]}})

    vec = asyncio.run(make(handler, model="gemini-embedding-2", dims=2).embed("hello"))
    assert seen["url"].endswith("/gemini-embedding-2:embedContent")
    assert seen["key"] == "secret-key" and "secret-key" not in seen["url"]
    assert seen["body"]["content"]["parts"][0]["text"] == "hello"
    assert seen["body"]["outputDimensionality"] == 2
    assert np.allclose(vec, [0.6, 0.8]) and abs(float(np.linalg.norm(vec)) - 1) < 1e-6


def test_gemini_embedder_raises_on_http_error():
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(make(lambda r: httpx.Response(429, json={"error": "quota"})).embed("x"))
