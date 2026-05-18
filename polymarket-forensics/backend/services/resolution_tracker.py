"""Attribute resolved markets back to the trades they cover.

For each newly-resolved market:
  1. Update the market row with status + outcome + resolved_at.
  2. For every trade in that market that hasn't been attributed yet,
     compute win/loss, PnL (net of fees), and hours-before-resolution.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

from sqlalchemy import select, update

from config import get_settings
from data.database import db_session
from data.models import Market, Trade
from data.polymarket_client import get_polymarket
from exceptions import ExternalAPIError
from utils.logging import get_logger
from utils.time import to_utc

log = get_logger(__name__)
settings = get_settings()

# Polymarket fee: 2% on winning side
WINNING_FEE = Decimal("0.02")
ZERO = Decimal("0")
ONE = Decimal("1")
EPS = Decimal("0.000001")


async def _refresh_markets() -> int:
    poly = await get_polymarket()
    try:
        markets = await poly.list_markets(limit=200, active=False)
    except ExternalAPIError as exc:
        log.warning("resolution_markets_failed", status=exc.status)
        return 0
    n = 0
    async with db_session() as session:
        for m in markets:
            mid = str(m.get("id") or m.get("conditionId") or "")
            if not mid:
                continue
            outcome = m.get("resolvedOutcome")
            if not m.get("closed") or not outcome:
                continue
            resolved_at = to_utc(
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


def _compute_pnl(side: str, outcome: str, resolution: str,
                 size: Decimal, price: Decimal) -> tuple[bool, Decimal]:
    """Return (trade_won, net_pnl)."""
    won = (side.upper() == "BUY" and outcome.upper() == resolution.upper())
    if won:
        gross = size * (ONE - price) / max(price, EPS)
        net = gross * (ONE - WINNING_FEE)
        return True, net
    return False, -size


async def _attribute_trades() -> int:
    async with db_session() as session:
        res = await session.execute(
            select(Market).where(
                (Market.status == "resolved")
                & Market.resolution_outcome.is_not(None)
            )
        )
        markets = list(res.scalars())

    total = 0
    for market in markets:
        async with db_session() as session:
            trades_res = await session.execute(
                select(Trade).where(
                    (Trade.market_id == market.id)
                    & (Trade.trade_won.is_(None))
                )
            )
            trades = list(trades_res.scalars())
            for t in trades:
                won, pnl = _compute_pnl(
                    t.side, t.outcome, market.resolution_outcome or "",
                    Decimal(t.size), Decimal(t.price),
                )
                hbr: Decimal | None = None
                if market.resolved_at and t.timestamp:
                    delta = market.resolved_at - t.timestamp
                    hbr = Decimal(max(0, delta.total_seconds())) / Decimal("3600")
                await session.execute(
                    update(Trade).where(Trade.id == t.id).values(
                        resolution_outcome=market.resolution_outcome,
                        trade_won=won,
                        pnl=pnl,
                        hours_before_resolution=hbr,
                    )
                )
                total += 1
    return total


async def run_once() -> dict[str, int]:
    refreshed = await _refresh_markets()
    attributed = await _attribute_trades()
    stats = {"resolved_markets": refreshed, "trades_attributed": attributed}
    log.info("resolution_done", **stats)
    return stats


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.resolution_interval
    log.info("resolution_loop_start", interval_sec=interval)
    while not stop_event.is_set():
        try:
            await run_once()
        except Exception:
            log.exception("resolution_loop_error")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            pass
