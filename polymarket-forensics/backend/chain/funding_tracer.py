"""Recursive funding-chain tracer.

Walks backwards from a target wallet to find original sources of capital
— exchanges, bridges, or other wallets. Stops at exchanges/bridges/contracts
or when depth limit is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation

from chain.address_classifier import classify_address
from chain.alchemy_client import AlchemyClient
from enums import SourceType
from exceptions import ExternalAPIError
from utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class FundingHop:
    """One edge of the funding tree."""

    wallet_address: str        # target wallet the trace started from
    source_address: str
    source_type: SourceType
    source_exchange: str | None
    amount: Decimal
    asset: str
    timestamp: datetime | None
    tx_hash: str
    depth: int


def _parse_amount(value: object) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        log.debug("funding_tracer_bad_amount", value=value)
        return Decimal("0")


def _parse_ts(meta: object) -> datetime | None:
    if not isinstance(meta, dict):
        return None
    raw = meta.get("blockTimestamp")
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(UTC)
    except (TypeError, ValueError):
        log.debug("funding_tracer_bad_timestamp", value=raw)
        return None


async def trace_funding_chain(
    wallet_address: str,
    alchemy: AlchemyClient,
    *,
    max_depth: int = 3,
    max_per_hop: int = 25,
) -> list[FundingHop]:
    """BFS-style trace of incoming funds.

    Args:
        wallet_address: 0x-prefixed Polygon address. Lower-cased internally.
        alchemy:        Alchemy client instance.
        max_depth:      How many hops backwards to walk before stopping.
        max_per_hop:    Cap on transfers fetched at each address.

    Returns:
        List of ``FundingHop`` records, one per incoming transfer discovered
        across all walks. Empty list if the wallet has no incoming USDC/MATIC.

    Notes:
        - Exchanges, bridges, and contracts are terminal — we never recurse
          past them (they aren't user-controlled).
        - A wallet seen more than once is skipped to avoid cycles.
    """
    target = wallet_address.lower()
    visited: set[str] = set()
    hops: list[FundingHop] = []
    queue: list[tuple[str, int]] = [(target, 0)]

    while queue:
        addr, depth = queue.pop(0)
        if depth > max_depth or addr in visited:
            continue
        visited.add(addr)

        try:
            transfers = await alchemy.get_asset_transfers(
                to_address=addr,
                category=["erc20", "external"],
                order="desc",
                max_count=max_per_hop,
            )
        except ExternalAPIError as exc:
            log.warning("funding_tracer_alchemy_error",
                        address=addr, depth=depth,
                        service=exc.service, status=exc.status)
            continue

        for tx in transfers:
            src = (tx.get("from") or "").lower()
            if not src:
                continue
            classification = await classify_address(src, alchemy)
            stype = SourceType(classification["type"])
            sexc = classification["exchange"] or None

            hops.append(FundingHop(
                wallet_address=target,
                source_address=src,
                source_type=stype,
                source_exchange=sexc,
                amount=_parse_amount(tx.get("value")),
                asset=(tx.get("asset") or "").upper() or "UNKNOWN",
                timestamp=_parse_ts(tx.get("metadata")),
                tx_hash=tx.get("hash", ""),
                depth=depth,
            ))

            if stype == SourceType.WALLET and depth < max_depth:
                queue.append((src, depth + 1))

    return hops
