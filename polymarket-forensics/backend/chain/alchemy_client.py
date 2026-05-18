"""
Alchemy JSON-RPC client for Polygon. Used by the funding-chain tracer.
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


class AlchemyClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.alchemy_api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._req_id = 0

    @property
    def url(self) -> str:
        return f"{settings.alchemy_polygon_url}/{self.api_key}"

    async def _ensure(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=5.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
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
    async def _rpc(self, method: str, params: list[Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("ALCHEMY_API_KEY is not configured")
        client = await self._ensure()
        self._req_id += 1
        body = {
            "jsonrpc": "2.0",
            "id": self._req_id,
            "method": method,
            "params": params,
        }
        r = await client.post(self.url, json=body)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            raise RuntimeError(f"Alchemy RPC error: {data['error']}")
        return data.get("result")

    async def get_asset_transfers(
        self,
        to_address: str | None = None,
        from_address: str | None = None,
        category: list[str] | None = None,
        max_count: int = 50,
        order: str = "desc",
    ) -> list[dict]:
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
        transfers = (result or {}).get("transfers", [])
        return transfers

    async def get_code(self, address: str) -> str:
        return await self._rpc("eth_getCode", [address, "latest"]) or "0x"

    async def get_tx_count(self, address: str) -> int:
        hex_count = await self._rpc("eth_getTransactionCount", [address, "latest"])
        return int(hex_count, 16) if hex_count else 0


_singleton: Optional[AlchemyClient] = None
_lock = asyncio.Lock()


async def get_alchemy() -> AlchemyClient:
    global _singleton
    async with _lock:
        if _singleton is None:
            _singleton = AlchemyClient()
    return _singleton
