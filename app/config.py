import os
from dataclasses import dataclass, field


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass
class Settings:
    api_keys: list[str] = field(default_factory=lambda: ["dev-key-1"])
    gemini_api_key: str = ""
    provider_order: list[str] = field(default_factory=lambda: ["gemini", "mock"])
    rate_limit_rpm: int = 60
    rate_limit_burst: int = 10
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000
    db_path: str = "gateway.db"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            api_keys=_csv(os.getenv("GATEWAY_API_KEYS", "dev-key-1")),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            provider_order=_csv(os.getenv("PROVIDER_ORDER", "gemini,mock")),
            rate_limit_rpm=int(os.getenv("RATE_LIMIT_RPM", "60")),
            rate_limit_burst=int(os.getenv("RATE_LIMIT_BURST", "10")),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "300")),
            cache_max_entries=int(os.getenv("CACHE_MAX_ENTRIES", "1000")),
            db_path=os.getenv("DB_PATH", "gateway.db"),
        )
