"""
Recomputes insider/smart scores for wallets with recent trade activity.
Pushes alerts for threshold crossings.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from sqlalchemy import select, update

from api.websocket import manager
from config import get_settings
from data.database import db_session
from data.models import Alert, Cluster, Trade, Wallet
from scoring.insider_score import (
    WalletProfile, compute_insider_score, classification_from_score,
)
from scoring.smart_score import compute_smart_score
from utils.logging import get_logger

log = get_logger("scoring")
settings = get_settings()

POLITICAL_PREFIXES = tuple(settings.political_categories)


def _is_political(category: str | None) -> bool:
    if not category:
        return False
    c = category.lower()
    return any(p in c for p in POLITICAL_PREFIXES)


async def _build_profile(session, address: str) -> WalletProfile | None:
    res = await session.execute(select(Wallet).where(Wallet.address == address))
    wallet = res.scalar_one_or_none()
    if wallet is None:
        return None

    trades_res = await session.execute(
        select(Trade).where(Trade.wallet_address == address)
    )
    trades = list(trades_res.scalars())
    if not trades:
        return None

    total_volume = sum(float(t.size) for t in trades)
    total_trades = len(trades)
    markets = {t.market_id for t in trades}

    per_market: dict[str, float] = defaultdict(float)
    per_category: dict[str, float] = defaultdict(float)
    political_vol = 0.0
    for t in trades:
        per_market[t.market_id] += float(t.size)
        if t.market_category:
            per_category[t.market_category] += float(t.size)
        if _is_political(t.market_category):
            political_vol += float(t.size)

    top_market_vol = max(per_market.values()) if per_market else 0.0
    top_category_vol = max(per_category.values()) if per_category else 0.0

    resolved = [t for t in trades if t.trade_won is not None]
    wins = sum(1 for t in resolved if t.trade_won)
    total_pnl = sum(float(t.pnl or 0) for t in trades)
    avg_entry = (sum(float(t.price) for t in trades) / total_trades) if total_trades else 0.0
    avg_size = (total_volume / total_trades) if total_trades else 0.0
    avg_pnl = (total_pnl / total_trades) if total_trades else 0.0

    hours_vals = [float(t.hours_before_resolution)
                  for t in trades if t.hours_before_resolution is not None]
    avg_hours = (sum(hours_vals) / len(hours_vals)) if hours_vals else None

    first_trade_ts = min(t.timestamp for t in trades)

    cluster_prob = 0.0
    if wallet.cluster_id:
        cres = await session.execute(
            select(Cluster.insider_probability).where(Cluster.id == wallet.cluster_id)
        )
        cp = cres.scalar_one_or_none()
        if cp is not None:
            cluster_prob = float(cp)

    return WalletProfile(
        address=address,
        first_seen=wallet.first_seen or first_trade_ts,
        first_trade=first_trade_ts,
        total_trades=total_trades,
        total_volume=total_volume,
        total_pnl=total_pnl,
        wins=wins,
        total_resolved=len(resolved),
        avg_entry_price=avg_entry,
        avg_trade_size=avg_size,
        avg_pnl_per_trade=avg_pnl,
        avg_hours_before_resolution=avg_hours,
        top_market_volume=top_market_vol,
        top_category_volume=top_category_vol,
        political_military_volume=political_vol,
        unique_protocols=wallet.unique_protocols or 0,
        total_tx_count=wallet.total_tx_count or 0,
        markets_traded=len(markets),
        cluster_id=wallet.cluster_id,
        cluster_insider_prob=cluster_prob,
    )


async def _emit_alert(
    session, alert_type: str, severity: str, title: str,
    description: str, wallet_address: str | None = None,
    data: dict | None = None,
) -> None:
    alert = Alert(
        alert_type=alert_type,
        severity=severity,
        title=title,
        description=description,
        wallet_address=wallet_address,
        data=data or {},
    )
    session.add(alert)
    await session.flush()
    await manager.broadcast({
        "type": "alert",
        "id": alert.id,
        "alert_type": alert_type,
        "severity": severity,
        "title": title,
        "description": description,
        "wallet_address": wallet_address,
        "data": data or {},
    })


async def score_wallet(address: str) -> dict | None:
    async with db_session() as session:
        profile = await _build_profile(session, address)
        if profile is None:
            return None

        result = compute_insider_score(profile)
        smart = compute_smart_score(profile)
        classification = classification_from_score(result["insider_score"])

        wallet_res = await session.execute(select(Wallet).where(Wallet.address == address))
        wallet = wallet_res.scalar_one()
        prev_score = float(wallet.insider_score or 0)

        await session.execute(
            update(Wallet)
            .where(Wallet.address == address)
            .values(
                insider_score=Decimal(str(result["insider_score"])),
                smart_score=Decimal(str(smart)),
                score_breakdown=result["breakdown"],
                win_rate_p_value=Decimal(str(result["win_rate_p_value"])),
                total_resolved=profile.total_resolved,
                wins=profile.wins,
                win_rate=Decimal(str(profile.wins / profile.total_resolved))
                    if profile.total_resolved else Decimal(0),
                markets_traded=profile.markets_traded,
                total_trades=profile.total_trades,
                total_volume=Decimal(str(profile.total_volume)),
                total_pnl=Decimal(str(profile.total_pnl)),
                avg_entry_price=Decimal(str(profile.avg_entry_price))
                    if profile.avg_entry_price else None,
                avg_trade_size=Decimal(str(profile.avg_trade_size))
                    if profile.avg_trade_size else None,
                avg_hours_before_resolution=Decimal(str(profile.avg_hours_before_resolution))
                    if profile.avg_hours_before_resolution else None,
                top_market_volume=Decimal(str(profile.top_market_volume)),
                top_category_volume=Decimal(str(profile.top_category_volume)),
                political_military_volume=Decimal(str(profile.political_military_volume)),
                classification=classification,
            )
        )

        new_score = result["insider_score"]
        if prev_score < 0.7 <= new_score:
            await _emit_alert(
                session,
                alert_type="INSIDER_SCORE_SPIKE",
                severity="high",
                title=f"Wallet crossed insider threshold: {address[:10]}...",
                description=f"Score {prev_score:.2f} → {new_score:.2f}. "
                            f"Class: {classification}. "
                            f"Win rate p-value: {result['win_rate_p_value']:.2e}.",
                wallet_address=address,
                data={"score": new_score, "breakdown": result["breakdown"]},
            )

        if (profile.total_resolved >= 10
                and result["win_rate_p_value"] < 0.001
                and prev_score < 0.7):
            await _emit_alert(
                session,
                alert_type="IMPOSSIBLE_WIN_RATE",
                severity="critical",
                title=f"Impossible win rate: {address[:10]}...",
                description=f"{profile.wins}/{profile.total_resolved} wins at "
                            f"avg entry {profile.avg_entry_price:.2f}. "
                            f"P-value: {result['win_rate_p_value']:.2e}.",
                wallet_address=address,
                data={
                    "wins": profile.wins,
                    "total": profile.total_resolved,
                    "p_value": result["win_rate_p_value"],
                    "pnl": profile.total_pnl,
                },
            )

        if profile.total_volume > 0 and (profile.top_market_volume / profile.total_volume) >= 0.9 \
                and profile.markets_traded <= 2 and profile.total_volume >= 5000:
            await _emit_alert(
                session,
                alert_type="SINGLE_MARKET_ALL_IN",
                severity="high",
                title=f"All-in concentration: {address[:10]}...",
                description=f"{(profile.top_market_volume / profile.total_volume) * 100:.0f}% "
                            f"of ${profile.total_volume:,.0f} in one market.",
                wallet_address=address,
                data={"top_market_pct": profile.top_market_volume / profile.total_volume},
            )

        return {
            "address": address,
            "insider_score": new_score,
            "smart_score": smart,
            "classification": classification,
            "breakdown": result["breakdown"],
        }


async def score_recent() -> int:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(minutes=10)
    async with db_session() as session:
        res = await session.execute(
            select(Wallet.address).where(Wallet.last_active >= cutoff)
        )
        addresses = [row[0] for row in res.all()]
    log.info("scoring %d recently-active wallets", len(addresses))
    n = 0
    for addr in addresses:
        try:
            await score_wallet(addr)
            n += 1
        except Exception as exc:
            log.warning("scoring failed for %s: %s", addr, exc)
    return n


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.scoring_interval
    while not stop_event.is_set():
        try:
            n = await score_recent()
            log.info("scored %d wallets", n)
        except Exception as exc:
            log.exception("scoring loop error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
