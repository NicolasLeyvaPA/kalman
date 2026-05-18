from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/forensics"
    alchemy_api_key: str = ""

    trade_poll_interval: int = 60
    scoring_interval: int = 300
    cluster_interval: int = 900
    resolution_interval: int = 3600

    large_trade_usd: float = 5000.0
    insider_trace_threshold: float = 0.3

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_origin: str = "http://localhost:5173"

    polymarket_gamma_url: str = "https://gamma-api.polymarket.com"
    polymarket_data_url: str = "https://data-api.polymarket.com"
    polymarket_clob_url: str = "https://clob.polymarket.com"

    alchemy_polygon_url: str = "https://polygon-mainnet.g.alchemy.com/v2"

    political_categories: list[str] = [
        "politics", "elections", "geopolitics", "military",
        "war", "ukraine", "russia", "iran", "israel", "china",
        "trump", "biden", "supreme-court", "congress",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()
