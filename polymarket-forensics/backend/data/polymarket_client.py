"""HTTP client for Polymarket's public APIs (Gamma / Data / CLOB).

Every method raises a typed ``PolymarketAPIError`` (or its ``RateLimitError``
subclass) on non-success responses. Callers decide whether to swallow,
retry, or propagate.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_settings
from exceptions import (
    PolymarketAPIError,
    RateLimitError,
    UpstreamUnavailableError,
)
from utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class _RateBucket:
    """Token bucket rate limiter — simple but effective."""

    def __init__(self, rate_per_sec: float, burst: int) -> None:
        self.rate = rate_per_sec
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.updated = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            loop = asyncio.get_event_loop()
            now = loop.time()
            elapsed = now - self.updated if self.updated else 0
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens < 1:
                wait = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait)
                self.tokens = 0
            else:
                self.tokens -= 1


class PolymarketClient:
    """Thin async wrapper around the Polymarket public APIs."""

    def __init__(
        self,
        *,
        gamma_url: str | None = None,
        data_url: str | None = None,
        clob_url: str | None = None,
        rate_per_sec: float = 5.0,
        burst: int = 10,
        timeout_sec: float = 20.0,
    ) -> None:
        self.gamma_url = gamma_url or settings.polymarket_gamma_url
        self.data_url = data_url or settings.polymarket_data_url
        self.clob_url = clob_url or settings.polymarket_clob_url
        self._client: httpx.AsyncClient | None = None
        self._rate = _RateBucket(rate_per_sec, burst)
        self._timeout = timeout_sec

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers={"User-Agent": "polymarket-forensics/1.0"},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((UpstreamUnavailableError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _get(
        self, url: str, params: dict[str, Any] | None = None, *, allow_404: bool = False,
    ) -> Any:
        await self._rate.acquire()
        client = await self._ensure()
        try:
            response = await client.get(url, params=params)
        except httpx.TimeoutException as exc:
            log.warning("polymarket_timeout", url=url, error=str(exc))
            raise UpstreamUnavailableError(
                "polymarket", url, 0, "timeout", context={"params": params},
            ) from exc
        except httpx.NetworkError as exc:
            log.warning("polymarket_network", url=url, error=str(exc))
            raise UpstreamUnavailableError(
                "polymarket", url, 0, str(exc), context={"params": params},
            ) from exc

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "60"))
            log.warning("polymarket_rate_limited", url=url, retry_after=retry_after)
            raise RateLimitError("polymarket", url, retry_after, response.text)
        if response.status_code == 404 and allow_404:
            return None
        if response.status_code >= 500:
            raise UpstreamUnavailableError(
                "polymarket", url, response.status_code, response.text,
                context={"params": params},
            )
        if response.status_code >= 400:
            raise PolymarketAPIError(
                url, response.status_code, response.text, context={"params": params},
            )
        return response.json()

    @staticmethod
    def _to_list(payload: Any) -> list[dict[str, Any]]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("data") or []
        return []

    # --- discovery -------------------------------------------------------

    async def list_markets(
        self, *, limit: int = 100, offset: int = 0, active: bool = True,
    ) -> list[dict[str, Any]]:
        url = f"{self.gamma_url}/markets"
        params = {"limit": limit, "offset": offset, "active": str(active).lower()}
        return self._to_list(await self._get(url, params=params))

    async def get_market(self, market_id: str) -> dict[str, Any] | None:
        url = f"{self.gamma_url}/markets/{market_id}"
        result = await self._get(url, allow_404=True)
        return result if isinstance(result, dict) else None

    async def list_events(
        self, *, limit: int = 100, offset: int = 0,
    ) -> list[dict[str, Any]]:
        url = f"{self.gamma_url}/events"
        params = {"limit": limit, "offset": offset}
        return self._to_list(await self._get(url, params=params))

    # --- trades / activity ----------------------------------------------

    async def get_market_trades(
        self, market_id: str, *, limit: int = 500,
    ) -> list[dict[str, Any]]:
        url = f"{self.data_url}/trades"
        params = {"market": market_id, "limit": limit}
        try:
            return self._to_list(await self._get(url, params=params, allow_404=True))
        except (PolymarketAPIError, RateLimitError) as exc:
            log.warning("polymarket_trades_error", market=market_id, status=exc.status)
            return []

    async def get_user_activity(
        self, address: str, *, limit: int = 200,
    ) -> list[dict[str, Any]]:
        url = f"{self.data_url}/activity"
        params = {"user": address, "limit": limit}
        try:
            return self._to_list(await self._get(url, params=params, allow_404=True))
        except (PolymarketAPIError, RateLimitError) as exc:
            log.warning("polymarket_activity_error", user=address, status=exc.status)
            return []

    async def get_user_positions(
        self, address: str, *, closed: bool = False,
    ) -> list[dict[str, Any]]:
        url = f"{self.data_url}/positions"
        params: dict[str, Any] = {"user": address}
        if closed:
            params["closed"] = "true"
        try:
            return self._to_list(await self._get(url, params=params, allow_404=True))
        except (PolymarketAPIError, RateLimitError) as exc:
            log.warning("polymarket_positions_error", user=address, status=exc.status)
            return []

    async def get_top_holders(self, market_id: str) -> list[dict[str, Any]]:
        url = f"{self.data_url}/top-holders"
        params = {"market": market_id}
        try:
            return self._to_list(await self._get(url, params=params, allow_404=True))
        except (PolymarketAPIError, RateLimitError):
            return []

    async def get_leaderboard(self) -> list[dict[str, Any]]:
        url = f"{self.data_url}/leaderboard"
        try:
            return self._to_list(await self._get(url, allow_404=True))
        except (PolymarketAPIError, RateLimitError):
            return []

    async def get_user_value(self, address: str) -> dict[str, Any]:
        url = f"{self.data_url}/value"
        params = {"user": address}
        try:
            result = await self._get(url, params=params, allow_404=True)
            return result if isinstance(result, dict) else {}
        except (PolymarketAPIError, RateLimitError):
            return {}

    # --- CLOB pricing ---------------------------------------------------

    async def get_midpoint(self, token_id: str) -> dict[str, Any]:
        url = f"{self.clob_url}/midpoint"
        try:
            result = await self._get(url, params={"token_id": token_id}, allow_404=True)
            return result if isinstance(result, dict) else {}
        except (PolymarketAPIError, RateLimitError):
            return {}

    async def get_book(self, token_id: str) -> dict[str, Any]:
        url = f"{self.clob_url}/book"
        try:
            result = await self._get(url, params={"token_id": token_id}, allow_404=True)
            return result if isinstance(result, dict) else {}
        except (PolymarketAPIError, RateLimitError):
            return {}


_singleton: PolymarketClient | None = None
_lock = asyncio.Lock()


async def get_polymarket() -> PolymarketClient:
    """Process-wide singleton client. Use DI in tests instead."""
    global _singleton
    async with _lock:
        if _singleton is None:
            _singleton = PolymarketClient()
    return _singleton
