"""
For all recently-resolved markets, attribute trades to win/loss and PnL.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import select, update

from config import get_settings
from data.database import db_session
from data.models import Market, Trade
from data.polymarket_client import get_polymarket
from utils.logging import get_logger


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
            return None
    return None

log = get_logger("resolution")
settings = get_settings()


async def _refresh_markets() -> int:
    poly = await get_polymarket()
    markets = await poly.list_markets(limit=200, active=False)
    n = 0
    async with db_session() as session:
        for m in markets:
            mid = str(m.get("id") or m.get("conditionId") or "")
            if not mid:
                continue
            outcome = m.get("resolvedOutcome")
            if not m.get("closed") or not outcome:
                continue
            resolved_at = _to_dt(
                m.get("resolvedDate") or m.get("endDate") or m.get("updatedAt")
            )
            await session.execute(
                update(Market).where(Market.id == mid).values(
                    status="resolved",
                    resolution_outcome=str(outcome).upper(),
                    resolved_at=resolved_at,
                )
            )
            n += 1
    return n


async def _attribute_trades() -> int:
    async with db_session() as session:
        res = await session.execute(
            select(Market).where(
                (Market.status == "resolved") & Market.resolution_outcome.is_not(None)
            )
        )
        markets = list(res.scalars())

    total = 0
    for market in markets:
        async with db_session() as session:
            trades_res = await session.execute(
                select(Trade).where(
                    (Trade.market_id == market.id) & (Trade.trade_won.is_(None))
                )
            )
            trades = list(trades_res.scalars())
            for t in trades:
                won = (t.side.upper() == "BUY"
                       and t.outcome.upper() == (market.resolution_outcome or "").upper())
                price = float(t.price)
                size = float(t.size)
                if won:
                    pnl = size * (1.0 - price) / max(price, 1e-6)
                else:
                    pnl = -size

                hbr = None
                if market.resolved_at and t.timestamp:
                    hbr = max(0.0, (market.resolved_at - t.timestamp).total_seconds() / 3600.0)

                await session.execute(
                    update(Trade).where(Trade.id == t.id).values(
                        resolution_outcome=market.resolution_outcome,
                        trade_won=won,
                        pnl=Decimal(str(pnl)),
                        hours_before_resolution=Decimal(str(hbr)) if hbr is not None else None,
                    )
                )
                total += 1
    return total


async def run_once() -> dict[str, int]:
    refreshed = await _refresh_markets()
    attributed = await _attribute_trades()
    return {"resolved_markets": refreshed, "trades_attributed": attributed}


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.resolution_interval
    while not stop_event.is_set():
        try:
            stats = await run_once()
            log.info("resolution: %s", stats)
        except Exception as exc:
            log.exception("resolution loop error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
