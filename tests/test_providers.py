"""Provider adapters, exercised against httpx.MockTransport (no network)."""
import asyncio
import json

import httpx
import pytest

from app.providers import GeminiProvider, OpenAICompatProvider, ProviderError
from app.schemas import ChatRequest, Message

REQ = ChatRequest(
    model="gemini-3.5-flash",
    messages=[
        Message(role="system", content="Be brief."),
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello!"),
        Message(role="user", content="Capital of France?"),
    ],
    temperature=0,
    max_tokens=50,
)


def gemini(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiProvider("k", client=client, **kw)


def openai_like(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatProvider("groq", "https://api.example.com/v1", "sk-x", client=client, **kw)


def sse(*events: str) -> bytes:
    return "".join(f"data: {e}\n\n" for e in events).encode()


def test_gemini_translates_request_and_parses_response():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers["x-goog-api-key"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Paris"}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 1},
        })

    resp = asyncio.run(gemini(handler).chat(REQ))
    body = seen["body"]
    assert seen["url"].endswith("/gemini-3.5-flash:generateContent") and seen["key"] == "k"
    assert body["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"] == {"temperature": 0, "maxOutputTokens": 50}
    assert resp.choices[0].message.content == "Paris"
    assert resp.usage.total_tokens == 13


@pytest.mark.parametrize("status,retryable", [(503, True), (429, True), (500, True),
                                              (404, False), (400, False), (401, False)])
def test_gemini_http_errors_are_classified(status, retryable):
    provider = gemini(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.chat(REQ))
    assert exc.value.retryable is retryable


def test_gemini_timeout_is_retryable():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ProviderError) as exc:
        asyncio.run(gemini(handler).chat(REQ))
    assert exc.value.retryable is True


def test_gemini_empty_candidates_is_not_retryable():
    provider = gemini(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.chat(REQ))
    assert exc.value.retryable is False


def test_gemini_model_override():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "x"}]}}]})

    asyncio.run(gemini(handler, model_override="gemini-3.5-flash-lite").chat(REQ))
    assert "gemini-3.5-flash-lite:generateContent" in seen["url"]


def test_gemini_streaming_parses_sse():
    def handler(request):
        assert "streamGenerateContent" in str(request.url) and "alt=sse" in str(request.url)
        return httpx.Response(200, content=sse(
            json.dumps({"candidates": [{"content": {"parts": [{"text": "Par"}]}}]}),
            json.dumps({"candidates": [{"content": {"parts": [{"text": "is"}]}}],
                        "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}),
        ))

    async def go():
        return [e async for e in gemini(handler).stream(REQ)]

    events = asyncio.run(go())
    assert "".join(e.delta for e in events) == "Paris"
    assert events[-1].usage.total_tokens == 7


def test_gemini_stream_http_error_raises_before_any_event():
    async def go():
        return [e async for e in gemini(lambda r: httpx.Response(503, text="busy")).stream(REQ)]

    with pytest.raises(ProviderError) as exc:
        asyncio.run(go())
    assert exc.value.retryable is True


def test_openai_compat_request_and_model_override():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Paris"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10},
        })

    resp = asyncio.run(openai_like(handler, model_override="llama-x").chat(REQ))
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-x"
    assert seen["body"]["model"] == "llama-x"  # not the gemini name the client sent
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert resp.model == "llama-x" and resp.usage.total_tokens == 10


def test_openai_compat_streaming():
    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, content=sse(
            json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            json.dumps({"choices": [{"delta": {"content": "Par"}}]}),
            json.dumps({"choices": [{"delta": {"content": "is"}}]}),
            json.dumps({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}),
            "[DONE]",
        ))

    async def go():
        return [e async for e in openai_like(handler).stream(REQ)]

    events = asyncio.run(go())
    assert "".join(e.delta for e in events) == "Paris"
    assert events[-1].usage.completion_tokens == 2


@pytest.mark.parametrize("status,retryable", [(429, True), (503, True), (401, False)])
def test_openai_compat_errors_are_classified(status, retryable):
    provider = openai_like(lambda r: httpx.Response(status, text="err"))
    with pytest.raises(ProviderError) as exc:
        asyncio.run(provider.chat(REQ))
    assert exc.value.retryable is retryable
