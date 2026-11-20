import os
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.providers import MockProvider, Provider, ProviderError

AUTH = {"Authorization": "Bearer test-key"}
BODY = {"model": "gemini-3.5-flash", "messages": [{"role": "user", "content": "hi"}]}

REDIS_URL = os.getenv("REDIS_URL", "")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")

needs_redis = pytest.mark.skipif(not REDIS_URL, reason="set REDIS_URL to run Redis tests")
needs_postgres = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="set TEST_DATABASE_URL to run Postgres tests"
)


class FailingProvider(Provider):
    """Always fails. `retryable` controls whether the router should retry it."""

    def __init__(self, name="broken", retryable=True):
        self.name = name
        self.retryable = retryable
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        raise ProviderError("boom", retryable=self.retryable)


class CountingProvider(MockProvider):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    async def chat(self, request):
        self.calls += 1
        return await super().chat(request)


def make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        api_keys=["test-key"],
        database_url=f"sqlite:///{tmp_path / 't.db'}",
        retry_base_delay=0.001,
        retry_max_delay=0.002,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@contextmanager
def make_client(tmp_path, providers=None, embedder=None, **overrides):
    app = create_app(
        make_settings(tmp_path, **overrides),
        providers if providers is not None else [MockProvider()],
        embedder=embedder,
    )
    with TestClient(app) as client:  # `with` runs the lifespan and keeps one event loop
        yield client
