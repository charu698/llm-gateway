import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # --- auth / providers -------------------------------------------------
    api_keys: list[str] = field(default_factory=lambda: ["dev-key-1"])
    provider_order: list[str] = field(
        default_factory=lambda: ["gemini", "groq", "openrouter", "mock"]
    )
    gemini_api_key: str = ""
    gemini_model: str = ""  # optional: force this model whatever the client asks for
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"
    mock_latency_ms: int = 0       # simulated upstream latency (benchmarks)
    flaky_fail_rate: float = 0.0   # "flaky" provider fails this fraction of calls
    provider_timeout: float = 30.0

    # --- resilience ---------------------------------------------------------
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.2
    retry_max_delay: float = 2.0
    breaker_failure_threshold: int = 5
    breaker_recovery_seconds: float = 30.0

    # --- rate limit / cache ---------------------------------------------------
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000
    semantic_cache_enabled: bool = False
    semantic_threshold: float = 0.92
    semantic_embedder: str = "hashing"  # "hashing" (offline) or "gemini"
    semantic_embedding_model: str = "gemini-embedding-2"
    semantic_max_entries: int = 5000

    # --- storage --------------------------------------------------------------
    redis_url: str = ""  # empty = in-process cache + rate limiter
    database_url: str = "sqlite:///gateway.db"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()  # reads a local .env file if present
        e = os.getenv
        d = cls()
        return cls(
            api_keys=_csv(e("GATEWAY_API_KEYS", "dev-key-1")),
            provider_order=_csv(e("PROVIDER_ORDER", ",".join(d.provider_order))),
            gemini_api_key=e("GEMINI_API_KEY", ""),
            gemini_model=e("GEMINI_MODEL", ""),
            groq_api_key=e("GROQ_API_KEY", ""),
            groq_model=e("GROQ_MODEL", d.groq_model),
            openrouter_api_key=e("OPENROUTER_API_KEY", ""),
            openrouter_model=e("OPENROUTER_MODEL", d.openrouter_model),
            mock_latency_ms=int(e("MOCK_LATENCY_MS", "0")),
            flaky_fail_rate=float(e("FLAKY_FAIL_RATE", "0")),
            provider_timeout=float(e("PROVIDER_TIMEOUT", "30")),
            retry_max_attempts=int(e("RETRY_MAX_ATTEMPTS", "3")),
            retry_base_delay=float(e("RETRY_BASE_DELAY", "0.2")),
            retry_max_delay=float(e("RETRY_MAX_DELAY", "2.0")),
            breaker_failure_threshold=int(e("BREAKER_FAILURE_THRESHOLD", "5")),
            breaker_recovery_seconds=float(e("BREAKER_RECOVERY_SECONDS", "30")),
            rate_limit_rpm=int(e("RATE_LIMIT_RPM", "60")),
            rate_limit_burst=int(e("RATE_LIMIT_BURST", "10")),
            cache_ttl_seconds=int(e("CACHE_TTL_SECONDS", "300")),
            cache_max_entries=int(e("CACHE_MAX_ENTRIES", "1000")),
            semantic_cache_enabled=_bool(e("SEMANTIC_CACHE", "false")),
            semantic_threshold=float(e("SEMANTIC_THRESHOLD", str(d.semantic_threshold))),
            semantic_embedder=e("SEMANTIC_EMBEDDER", "hashing"),
            semantic_embedding_model=e("SEMANTIC_EMBEDDING_MODEL", d.semantic_embedding_model),
            semantic_max_entries=int(e("SEMANTIC_MAX_ENTRIES", "5000")),
            redis_url=e("REDIS_URL", ""),
            database_url=e("DATABASE_URL", "sqlite:///gateway.db"),
        )
