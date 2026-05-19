"""Alchemy Polygon JSON-RPC client."""

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
    AlchemyAPIError,
    AlchemyConfigError,
    RateLimitError,
    UpstreamUnavailableError,
)
from utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()


class AlchemyClient:
    """Thin async wrapper over Alchemy's Polygon endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_sec: float = 30.0,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.alchemy_key
        self.base_url = base_url or settings.alchemy_polygon_url
        self._client: httpx.AsyncClient | None = None
        self._req_id = 0
        self._timeout = timeout_sec

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.api_key}"

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout, connect=5.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((UpstreamUnavailableError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _rpc(self, method: str, params: list[Any]) -> Any:
        if not self.api_key:
            raise AlchemyConfigError(
                "ALCHEMY_API_KEY is not configured",
                context={"method": method},
            )
        client = await self._ensure()
        self._req_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }
        try:
            response = await client.post(self.url, json=body)
        except httpx.TimeoutException as exc:
            raise UpstreamUnavailableError(
                "alchemy", method, 0, "timeout", context={"params": params},
            ) from exc
        except httpx.NetworkError as exc:
            raise UpstreamUnavailableError(
                "alchemy", method, 0, str(exc), context={"params": params},
            ) from exc

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", "5"))
            raise RateLimitError("alchemy", method, retry_after, response.text)
        if response.status_code >= 500:
            raise UpstreamUnavailableError(
                "alchemy", method, response.status_code, response.text,
            )
        if response.status_code >= 400:
            raise AlchemyAPIError(
                method, response.status_code, response.text, context={"params": params},
            )

        data = response.json()
        if "error" in data:
            err = data["error"]
            code = int(err.get("code", -1))
            msg = err.get("message", "rpc error")
            # rate-limit codes from Alchemy
            if code in (-32005, -32016, 429):
                raise RateLimitError("alchemy", method, 5.0, str(err))
            raise AlchemyAPIError(method, 200, str(err), context={"code": code, "msg": msg})
        return data.get("result")

    async def get_asset_transfers(
        self,
        *,
        to_address: str | None = None,
        from_address: str | None = None,
        category: list[str] | None = None,
        max_count: int = 50,
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "category": category or ["erc20", "external"],
            "withMetadata": True,
            "excludeZeroValue": True,
            "maxCount": hex(max_count),
            "order": order,
        }
        if to_address:
            params["toAddress"] = to_address
        if from_address:
            params["fromAddress"] = from_address
        result = await self._rpc("alchemy_getAssetTransfers", [params])
        transfers = (result or {}).get("transfers", []) if isinstance(result, dict) else []
        return list(transfers)

    async def get_code(self, address: str) -> str:
        result = await self._rpc("eth_getCode", [address, "latest"])
        return str(result) if result else "0x"

    async def get_tx_count(self, address: str) -> int:
        hex_count = await self._rpc("eth_getTransactionCount", [address, "latest"])
        return int(str(hex_count), 16) if hex_count else 0


_singleton: AlchemyClient | None = None
_lock = asyncio.Lock()


async def get_alchemy() -> AlchemyClient:
    """Process-wide singleton client. Use DI in tests instead."""
    global _singleton
    async with _lock:
        if _singleton is None:
            _singleton = AlchemyClient()
    return _singleton
