"""
Polls Polymarket markets + trades endpoints. Upserts wallets and trades.
Triggers immediate scoring + funding trace for large trades.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from config import get_settings
from data.database import db_session
from data.models import Market, Trade, Wallet
from data.polymarket_client import get_polymarket
from utils.logging import get_logger

log = get_logger("ingester")
settings = get_settings()


def _to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except Exception:
                return None
    return None


def _normalize_trade(market: dict, t: dict) -> dict | None:
    addr = (t.get("proxyWallet") or t.get("maker") or t.get("user")
            or t.get("trader") or "").lower()
    if not addr:
        return None

    side = (t.get("side") or "").upper()
    outcome = (t.get("outcome") or "").upper()
    try:
        size = float(t.get("size") or t.get("amount") or t.get("usdcSize") or 0)
        price = float(t.get("price") or 0)
    except (TypeError, ValueError):
        return None
    if size <= 0 or price <= 0:
        return None

    ts = _to_dt(t.get("timestamp") or t.get("matchTime") or t.get("createdAt"))
    if ts is None:
        return None

    return {
        "wallet_address": addr,
        "market_id": str(market.get("id") or market.get("conditionId") or ""),
        "market_question": market.get("question") or market.get("slug"),
        "market_category": (market.get("category") or "").lower() or None,
        "side": side or ("BUY" if (t.get("type") or "").lower() == "buy" else "SELL"),
        "outcome": outcome or "YES",
        "size": Decimal(str(size)),
        "price": Decimal(str(price)),
        "timestamp": ts,
        "tx_hash": t.get("transactionHash") or t.get("txHash") or t.get("hash"),
        "is_large": size >= settings.large_trade_usd,
    }


async def _upsert_wallet(session, address: str, ts: datetime) -> None:
    stmt = insert(Wallet).values(
        address=address,
        first_seen=ts,
        last_active=ts,
    ).on_conflict_do_update(
        index_elements=[Wallet.address],
        set_={
            "last_active": ts,
        },
    )
    await session.execute(stmt)


async def _upsert_market(session, market: dict) -> None:
    mid = str(market.get("id") or market.get("conditionId") or "")
    if not mid:
        return
    stmt = insert(Market).values(
        id=mid,
        question=market.get("question") or market.get("slug") or mid,
        category=(market.get("category") or "").lower() or None,
        status="resolved" if market.get("closed") else "active",
        resolution_outcome=(market.get("resolvedOutcome") or None),
        current_price=Decimal(str(market.get("lastTradePrice") or market.get("price") or 0))
            if market.get("lastTradePrice") or market.get("price") else None,
        volume_total=Decimal(str(market.get("volume") or 0)) if market.get("volume") else None,
        created_at=_to_dt(market.get("createdAt")),
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


async def _insert_trade(session, trade: dict) -> bool:
    stmt = insert(Trade).values(**trade).on_conflict_do_nothing(
        index_elements=[Trade.wallet_address, Trade.market_id, Trade.tx_hash, Trade.timestamp]
    ).returning(Trade.id)
    res = await session.execute(stmt)
    return res.scalar_one_or_none() is not None


async def ingest_once() -> dict[str, int]:
    poly = await get_polymarket()
    markets = await poly.list_markets(limit=200, active=True)
    log.info("polled %d active markets", len(markets))

    new_trades = 0
    new_wallets = 0
    seen_addresses: set[str] = set()

    for market in markets[:80]:                        # respect rate limits
        mid = str(market.get("id") or market.get("conditionId") or "")
        if not mid:
            continue
        async with db_session() as session:
            await _upsert_market(session, market)

        trades = await poly.get_market_trades(mid, limit=200)
        for t in trades:
            payload = _normalize_trade(market, t)
            if not payload:
                continue
            async with db_session() as session:
                if payload["wallet_address"] not in seen_addresses:
                    await _upsert_wallet(session, payload["wallet_address"], payload["timestamp"])
                    seen_addresses.add(payload["wallet_address"])
                    new_wallets += 1
                inserted = await _insert_trade(session, payload)
                if inserted:
                    new_trades += 1

    return {"markets": len(markets), "trades": new_trades, "wallets_seen": len(seen_addresses)}


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.trade_poll_interval
    while not stop_event.is_set():
        try:
            stats = await ingest_once()
            log.info("ingest: %s", stats)
        except Exception as exc:
            log.exception("ingest loop error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
