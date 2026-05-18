"""
HTTP client for Polymarket's public APIs:
  - Gamma API   (markets/events)
  - Data API    (activity, positions, trades, leaderboard)
  - CLOB API    (order book, midpoint)
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import httpx
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type,
)

from config import get_settings

settings = get_settings()


class PolymarketClient:
    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(20.0, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"User-Agent": "PolymarketForensics/1.0"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def _get(self, url: str, params: dict | None = None) -> Any:
        client = await self._ensure()
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    # Markets / events
    async def list_markets(self, limit: int = 100, offset: int = 0,
                           active: bool = True) -> list[dict]:
        url = f"{settings.polymarket_gamma_url}/markets"
        params = {"limit": limit, "offset": offset, "active": str(active).lower()}
        data = await self._get(url, params=params)
        return data if isinstance(data, list) else data.get("data", [])

    async def get_market(self, market_id: str) -> dict:
        url = f"{settings.polymarket_gamma_url}/markets/{market_id}"
        return await self._get(url)

    async def list_events(self, limit: int = 100, offset: int = 0) -> list[dict]:
        url = f"{settings.polymarket_gamma_url}/events"
        params = {"limit": limit, "offset": offset}
        data = await self._get(url, params=params)
        return data if isinstance(data, list) else data.get("data", [])

    # Trades / activity
    async def get_market_trades(self, market_id: str, limit: int = 500) -> list[dict]:
        url = f"{settings.polymarket_data_url}/trades"
        params = {"market": market_id, "limit": limit}
        try:
            data = await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return []
        return data if isinstance(data, list) else data.get("data", [])

    async def get_user_activity(self, address: str, limit: int = 200) -> list[dict]:
        url = f"{settings.polymarket_data_url}/activity"
        params = {"user": address, "limit": limit}
        try:
            data = await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return []
        return data if isinstance(data, list) else data.get("data", [])

    async def get_user_positions(self, address: str, closed: bool = False) -> list[dict]:
        url = f"{settings.polymarket_data_url}/positions"
        params = {"user": address}
        if closed:
            params["closed"] = "true"
        try:
            data = await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return []
        return data if isinstance(data, list) else data.get("data", [])

    async def get_top_holders(self, market_id: str) -> list[dict]:
        url = f"{settings.polymarket_data_url}/top-holders"
        params = {"market": market_id}
        try:
            data = await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return []
        return data if isinstance(data, list) else data.get("data", [])

    async def get_leaderboard(self) -> list[dict]:
        url = f"{settings.polymarket_data_url}/leaderboard"
        try:
            data = await self._get(url)
        except httpx.HTTPStatusError:
            return []
        return data if isinstance(data, list) else data.get("data", [])

    async def get_user_value(self, address: str) -> dict:
        url = f"{settings.polymarket_data_url}/value"
        params = {"user": address}
        try:
            return await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return {}

    async def get_total_markets_traded(self, address: str) -> dict:
        url = f"{settings.polymarket_data_url}/total-markets-traded"
        params = {"user": address}
        try:
            return await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return {}

    # CLOB pricing
    async def get_midpoint(self, token_id: str) -> dict:
        url = f"{settings.polymarket_clob_url}/midpoint"
        params = {"token_id": token_id}
        try:
            return await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return {}

    async def get_book(self, token_id: str) -> dict:
        url = f"{settings.polymarket_clob_url}/book"
        params = {"token_id": token_id}
        try:
            return await self._get(url, params=params)
        except httpx.HTTPStatusError:
            return {}


_singleton: Optional[PolymarketClient] = None
_lock = asyncio.Lock()


async def get_polymarket() -> PolymarketClient:
    global _singleton
    async with _lock:
        if _singleton is None:
            _singleton = PolymarketClient()
    return _singleton
