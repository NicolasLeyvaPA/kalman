"""Poll Polymarket trades + markets, upsert wallets/markets/trades.

Triggers immediate scoring + funding-chain trace for fresh-whale activity.
Detects sensitive-market volume surges and emits alerts.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert

from config import get_settings
from data.database import db_session
from data.models import Market, Trade, Wallet
from data.polymarket_client import get_polymarket
from enums import AlertType, Severity
from exceptions import ExternalAPIError
from services import chain_tracer
from services.alert_generator import emit
from utils.logging import get_logger
from utils.time import to_utc

log = get_logger(__name__)
settings = get_settings()

POLITICAL = tuple(settings.political_categories)


def _is_political(category: str | None) -> bool:
    if not category:
        return False
    c = category.lower()
    return any(p in c for p in POLITICAL)


def _to_decimal(value: Any, default: str = "0") -> Decimal:
    if value is None:
        return Decimal(default)
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(default)


def _normalize_trade(market: dict[str, Any], t: dict[str, Any]) -> dict[str, Any] | None:
    addr = (
        t.get("proxyWallet") or t.get("maker") or t.get("user")
        or t.get("trader") or ""
    ).lower()
    if not addr:
        return None

    side = (t.get("side") or "").upper()
    outcome = (t.get("outcome") or "").upper()
    size = _to_decimal(t.get("size") or t.get("amount") or t.get("usdcSize"))
    price = _to_decimal(t.get("price"))
    if size <= 0 or price <= 0:
        return None

    ts = to_utc(t.get("timestamp") or t.get("matchTime") or t.get("createdAt"))
    if ts is None:
        return None

    if not side:
        side = "BUY" if (t.get("type") or "").lower() == "buy" else "SELL"

    return {
        "wallet_address": addr,
        "market_id": str(market.get("id") or market.get("conditionId") or ""),
        "market_question": market.get("question") or market.get("slug"),
        "market_category": (market.get("category") or "").lower() or None,
        "side": side,
        "outcome": outcome or "YES",
        "size": size,
        "price": price,
        "timestamp": ts,
        "tx_hash": t.get("transactionHash") or t.get("txHash") or t.get("hash"),
        "is_large": size >= settings.large_trade_usd,
    }


async def _upsert_wallet(session, address: str, ts: datetime) -> bool:
    """Insert-or-update. Returns True if the wallet was newly inserted."""
    stmt = insert(Wallet).values(
        address=address,
        first_seen=ts,
        last_active=ts,
    ).on_conflict_do_nothing(
        index_elements=[Wallet.address],
    ).returning(Wallet.address)
    res = await session.execute(stmt)
    inserted = res.scalar_one_or_none() is not None
    if not inserted:
        await session.execute(
            update(Wallet).where(Wallet.address == address).values(last_active=ts)
        )
    return inserted


async def _upsert_market(session, market: dict[str, Any]) -> None:
    mid = str(market.get("id") or market.get("conditionId") or "")
    if not mid:
        return
    stmt = insert(Market).values(
        id=mid,
        question=market.get("question") or market.get("slug") or mid,
        category=(market.get("category") or "").lower() or None,
        status="resolved" if market.get("closed") else "active",
        resolution_outcome=(market.get("resolvedOutcome") or None),
        current_price=_to_decimal(
            market.get("lastTradePrice") or market.get("price")
        ) or None,
        volume_total=_to_decimal(market.get("volume")) or None,
        created_at=to_utc(market.get("createdAt")),
    ).on_conflict_do_update(
        index_elements=[Market.id],
        set_={
            "question": market.get("question") or market.get("slug") or mid,
            "category": (market.get("category") or "").lower() or None,
            "status": "resolved" if market.get("closed") else "active",
            "resolution_outcome": market.get("resolvedOutcome") or None,
        },
    )
    await session.execute(stmt)


async def _insert_trade(session, trade: dict[str, Any]) -> bool:
    stmt = insert(Trade).values(**trade).on_conflict_do_nothing(
        index_elements=[
            Trade.wallet_address, Trade.market_id, Trade.tx_hash, Trade.timestamp,
        ]
    ).returning(Trade.id)
    res = await session.execute(stmt)
    return res.scalar_one_or_none() is not None


async def _maybe_alert_surge(
    session,
    market: dict[str, Any],
    new_trades_count: int,
    new_trades_volume: Decimal,
) -> None:
    """Fire SENSITIVE_MARKET_SURGE for political/military markets with sudden volume."""
    category = (market.get("category") or "").lower()
    if not _is_political(category):
        return
    if new_trades_volume < settings.large_trade_usd * Decimal("3"):
        return
    mid = str(market.get("id") or market.get("conditionId") or "")
    await emit(
        session,
        alert_type=AlertType.SENSITIVE_MARKET_SURGE,
        severity=Severity.CRITICAL,
        title=f"Sensitive-market volume surge ({category})",
        description=(
            f"{new_trades_count} trades totaling "
            f"${new_trades_volume:,.0f} on {market.get('question') or mid}."
        ),
        market_id=mid,
        data={"category": category, "new_volume": str(new_trades_volume),
              "new_trades": new_trades_count},
    )


async def ingest_once() -> dict[str, int]:
    """One ingest cycle. Returns summary stats."""
    poly = await get_polymarket()
    try:
        markets = await poly.list_markets(limit=200, active=True)
    except ExternalAPIError as exc:
        log.warning("ingest_markets_failed",
                    status=exc.status, service=exc.service)
        return {"markets": 0, "trades": 0, "wallets_new": 0}

    log.info("ingest_markets_fetched", count=len(markets))

    new_trades = 0
    new_wallets = 0
    seen_addresses: set[str] = set()
    per_market_new: dict[str, tuple[int, Decimal]] = defaultdict(
        lambda: (0, Decimal("0"))
    )

    for market in markets[: settings.ingest_max_markets]:
        mid = str(market.get("id") or market.get("conditionId") or "")
        if not mid:
            continue
        async with db_session() as session:
            await _upsert_market(session, market)

        try:
            trades = await poly.get_market_trades(mid, limit=200)
        except ExternalAPIError as exc:
            log.warning("ingest_trades_failed",
                        market=mid, status=exc.status, service=exc.service)
            continue

        async with db_session() as session:
            for t in trades:
                payload = _normalize_trade(market, t)
                if payload is None:
                    continue
                if payload["wallet_address"] not in seen_addresses:
                    is_new = await _upsert_wallet(
                        session, payload["wallet_address"], payload["timestamp"],
                    )
                    seen_addresses.add(payload["wallet_address"])
                    if is_new:
                        new_wallets += 1
                inserted = await _insert_trade(session, payload)
                if inserted:
                    new_trades += 1
                    count, vol = per_market_new[mid]
                    per_market_new[mid] = (count + 1, vol + payload["size"])
                    if payload["is_large"]:
                        await chain_tracer.enqueue(payload["wallet_address"])

            await _maybe_alert_surge(
                session, market, *per_market_new[mid],
            )

    log.info("ingest_done",
             markets=len(markets), new_trades=new_trades, new_wallets=new_wallets)
    return {"markets": len(markets), "trades": new_trades, "wallets_new": new_wallets}


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.trade_poll_interval
    log.info("ingest_loop_start", interval_sec=interval)
    while not stop_event.is_set():
        try:
            await ingest_once()
        except Exception:
            log.exception("ingest_loop_error")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            pass
