"""Application configuration via Pydantic Settings.

Every tunable lives here. Nothing is hardcoded elsewhere in the codebase.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- data sources ----------------------------------------------------
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/forensics"
    alchemy_api_key: SecretStr = SecretStr("")

    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_data_url: str = "https://data-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"
    alchemy_polygon_url: str = "https://polygon-mainnet.g.alchemy.com/v2"

    # --- service intervals (seconds) ------------------------------------
    trade_poll_interval: int = Field(60, gt=0)
    scoring_interval: int = Field(300, gt=0)
    cluster_interval: int = Field(900, gt=0)
    resolution_interval: int = Field(3600, gt=0)

    # --- alert thresholds -----------------------------------------------
    large_trade_usd: Decimal = Field(default=Decimal("5000"), gt=0)
    insider_trace_threshold: Decimal = Field(
        default=Decimal("0.3"), ge=Decimal("0"), le=Decimal("1"),
    )
    fresh_whale_max_age_days: int = Field(7, gt=0)
    fresh_whale_min_size_usd: Decimal = Field(default=Decimal("10000"), gt=0)
    sensitive_surge_ratio: Decimal = Field(default=Decimal("5"), gt=1)

    # --- ingest rate limiting -------------------------------------------
    polymarket_rate_per_sec: float = Field(5.0, gt=0)
    ingest_max_markets: int = Field(80, gt=0)

    # --- HTTP -----------------------------------------------------------
    backend_host: str = "0.0.0.0"
    backend_port: int = Field(8000, gt=0, lt=65536)
    frontend_origin: str = "http://localhost:5173"
    cors_extra_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # --- categorization --------------------------------------------------
    political_categories: tuple[str, ...] = Field(
        default=(
            "politics", "elections", "geopolitics", "military",
            "war", "ukraine", "russia", "iran", "israel", "china",
            "trump", "biden", "supreme-court", "congress",
        )
    )

    log_level: str = "INFO"

    @field_validator("political_categories", mode="before")
    @classmethod
    def _coerce_categories(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, str):
            return tuple(s.strip().lower() for s in v.split(",") if s.strip())
        if isinstance(v, (list, tuple)):
            return tuple(str(s).lower() for s in v)
        return ()

    @property
    def alchemy_key(self) -> str:
        return self.alchemy_api_key.get_secret_value() if self.alchemy_api_key else ""

    @property
    def cors_origins(self) -> list[str]:
        return [self.frontend_origin, *self.cors_extra_origins]


@lru_cache
def get_settings() -> Settings:
    return Settings()
