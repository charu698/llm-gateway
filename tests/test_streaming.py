import json

from app.providers import MockProvider
from tests.conftest import AUTH, BODY, FailingProvider, make_client

STREAM_BODY = {**BODY, "stream": True}


def read_sse(response) -> tuple[list[dict], bool]:
    chunks, done = [], False
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            done = True
        else:
            chunks.append(json.loads(payload))
    return chunks, done


def collect_text(chunks) -> str:
    return "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)


def test_streaming_response_shape(tmp_path):
    with make_client(tmp_path) as c:
        with c.stream("POST", "/v1/chat/completions", json=STREAM_BODY, headers=AUTH) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            assert r.headers["X-Gateway-Provider"] == "mock"
            chunks, done = read_sse(r)
    assert done
    assert all(ch["object"] == "chat.completion.chunk" for ch in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert collect_text(chunks) == "[mock] You said: hi"


def test_streamed_answer_is_cached_for_later_requests(tmp_path):
    with make_client(tmp_path) as c:
        with c.stream("POST", "/v1/chat/completions", json=STREAM_BODY, headers=AUTH) as r:
            streamed, _ = read_sse(r)
        plain = c.post("/v1/chat/completions", json=BODY, headers=AUTH)  # same prompt, no stream
    assert plain.headers["X-Cache"] == "HIT"
    assert plain.json()["choices"][0]["message"]["content"] == collect_text(streamed)


def test_cache_hit_replayed_as_stream(tmp_path):
    with make_client(tmp_path) as c:
        c.post("/v1/chat/completions", json=BODY, headers=AUTH)
        with c.stream("POST", "/v1/chat/completions", json=STREAM_BODY, headers=AUTH) as r:
            assert r.headers["X-Cache"] == "HIT"
            chunks, done = read_sse(r)
    assert done and collect_text(chunks) == "[mock] You said: hi"


def test_stream_fails_over_to_next_provider(tmp_path):
    with make_client(tmp_path, providers=[FailingProvider(), MockProvider()]) as c:
        with c.stream("POST", "/v1/chat/completions", json=STREAM_BODY, headers=AUTH) as r:
            assert r.status_code == 200 and r.headers["X-Gateway-Provider"] == "mock"
            chunks, done = read_sse(r)
    assert done and collect_text(chunks).startswith("[mock]")


def test_stream_all_providers_fail_returns_502(tmp_path):
    with make_client(tmp_path, providers=[FailingProvider()]) as c:
        r = c.post("/v1/chat/completions", json=STREAM_BODY, headers=AUTH)
    assert r.status_code == 502


def test_stream_stats_record_time_to_first_token(tmp_path):
    with make_client(tmp_path) as c:
        with c.stream("POST", "/v1/chat/completions", json=STREAM_BODY, headers=AUTH) as r:
            read_sse(r)
        s = c.get("/v1/stats", headers=AUTH).json()
    assert s["total_requests"] == 1 and s["p50_time_to_first_token_ms"] > 0
