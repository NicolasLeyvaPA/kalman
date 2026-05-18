"""
Recursive funding-chain tracer. Walks backwards from a wallet to find
the original sources of capital — exchanges, bridges, or other wallets.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from chain.alchemy_client import AlchemyClient
from chain.address_classifier import classify_address


@dataclass
class FundingHop:
    wallet_address: str          # target wallet (the root we started from)
    source_address: str
    source_type: str              # exchange | bridge | contract | wallet | unknown
    source_exchange: Optional[str]
    amount: float
    asset: str
    timestamp: Optional[datetime]
    tx_hash: str
    depth: int


def _parse_ts(meta: dict | None) -> Optional[datetime]:
    if not meta:
        return None
    bt = meta.get("blockTimestamp")
    if not bt:
        return None
    try:
        return datetime.fromisoformat(bt.replace("Z", "+00:00"))
    except Exception:
        return None


async def trace_funding_chain(
    wallet_address: str,
    alchemy: AlchemyClient,
    max_depth: int = 3,
    max_per_hop: int = 25,
) -> list[FundingHop]:
    """
    BFS-style trace. At each address we fetch incoming transfers, classify
    each source, and recurse into anything that looks like a regular wallet
    until we hit an exchange, a bridge, a contract, or max_depth.
    """
    target = wallet_address.lower()
    visited: set[str] = set()
    hops: list[FundingHop] = []

    async def walk(addr: str, depth: int) -> None:
        if depth > max_depth or addr in visited:
            return
        visited.add(addr)

        try:
            transfers = await alchemy.get_asset_transfers(
                to_address=addr,
                category=["erc20", "external"],
                order="desc",
                max_count=max_per_hop,
            )
        except Exception:
            return

        for tx in transfers:
            src = (tx.get("from") or "").lower()
            if not src:
                continue
            cls = await classify_address(src, alchemy)
            stype = cls["type"]
            sexc = cls["exchange"] or None

            value = tx.get("value")
            try:
                amount = float(value) if value is not None else 0.0
            except (TypeError, ValueError):
                amount = 0.0

            hops.append(FundingHop(
                wallet_address=target,
                source_address=src,
                source_type=stype,
                source_exchange=sexc,
                amount=amount,
                asset=(tx.get("asset") or "").upper() or "UNKNOWN",
                timestamp=_parse_ts(tx.get("metadata")),
                tx_hash=tx.get("hash", ""),
                depth=depth,
            ))

            if stype == "wallet" and depth < max_depth:
                await walk(src, depth + 1)

    await walk(target, 0)
    return hops
