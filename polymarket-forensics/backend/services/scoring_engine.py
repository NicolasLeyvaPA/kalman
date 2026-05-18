"""Score recently-active wallets and emit threshold-crossing alerts."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from data.database import db_session
from data.models import Cluster, Trade, Wallet
from enums import AlertType, Severity
from scoring.insider_score import compute_insider_score
from scoring.smart_score import compute_smart_score
from scoring.types import DEFAULT_CONFIG, InsiderScoreResult, ScoringConfig, WalletProfile
from services.alert_generator import emit
from utils.logging import get_logger
from utils.time import utc_now

log = get_logger(__name__)
settings = get_settings()

POLITICAL_PREFIXES = tuple(settings.political_categories)
ZERO = Decimal("0")
ONE = Decimal("1")


def _is_political(category: str | None) -> bool:
    if not category:
        return False
    c = category.lower()
    return any(p in c for p in POLITICAL_PREFIXES)


async def _build_profile(
    session: AsyncSession, address: str,
) -> tuple[Wallet, WalletProfile] | None:
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

    total_trades = len(trades)
    total_volume = sum((Decimal(t.size) for t in trades), ZERO)
    markets = {t.market_id for t in trades}

    per_market: dict[str, Decimal] = defaultdict(lambda: ZERO)
    per_category: dict[str, Decimal] = defaultdict(lambda: ZERO)
    political_vol = ZERO
    for t in trades:
        size = Decimal(t.size)
        per_market[t.market_id] += size
        if t.market_category:
            per_category[t.market_category] += size
        if _is_political(t.market_category):
            political_vol += size

    top_market_vol = max(per_market.values()) if per_market else ZERO
    top_category_vol = max(per_category.values()) if per_category else ZERO

    resolved = [t for t in trades if t.trade_won is not None]
    wins = sum(1 for t in resolved if t.trade_won)
    total_pnl = sum((Decimal(t.pnl or 0) for t in trades), ZERO)
    avg_entry = (
        sum((Decimal(t.price) for t in trades), ZERO) / Decimal(total_trades)
        if total_trades else ZERO
    )
    avg_size = (total_volume / Decimal(total_trades)) if total_trades else ZERO
    avg_pnl = (total_pnl / Decimal(total_trades)) if total_trades else ZERO

    hours_vals = [Decimal(t.hours_before_resolution)
                  for t in trades if t.hours_before_resolution is not None]
    avg_hours = (sum(hours_vals, ZERO) / Decimal(len(hours_vals))) if hours_vals else None

    first_trade_ts = min(t.timestamp for t in trades)

    cluster_prob = ZERO
    if wallet.cluster_id:
        cres = await session.execute(
            select(Cluster.insider_probability).where(Cluster.id == wallet.cluster_id)
        )
        cp = cres.scalar_one_or_none()
        if cp is not None:
            cluster_prob = Decimal(cp)

    profile = WalletProfile(
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
    return wallet, profile


async def _check_alerts(
    session: AsyncSession,
    wallet: Wallet,
    profile: WalletProfile,
    result: InsiderScoreResult,
    *,
    config: ScoringConfig,
    prev_score: Decimal,
) -> None:
    """Fire all threshold-based alerts for a freshly-scored wallet."""

    addr = profile.address

    # INSIDER_SCORE_SPIKE - first time we cross the 0.70 threshold.
    if prev_score < config.class_insider_suspect <= result.composite:
        await emit(
            session,
            alert_type=AlertType.INSIDER_SCORE_SPIKE,
            severity=Severity.HIGH,
            title=f"Insider score threshold crossed: {addr[:10]}...",
            description=(
                f"Score {prev_score:.2f} → {result.composite:.2f}. "
                f"Classification: {result.classification.value}. "
                f"Win-rate p-value: {result.win_rate_p_value:.2e}."
            ),
            wallet_address=addr,
            data={
                "previous_score": str(prev_score),
                "new_score": str(result.composite),
                "breakdown": result.breakdown.as_dict(),
            },
        )

    # IMPOSSIBLE_WIN_RATE - p < 0.001 with at least N resolved trades.
    if (profile.total_resolved >= config.impossible_winrate_min_resolved
            and result.win_rate_p_value < config.p_high):
        await emit(
            session,
            alert_type=AlertType.IMPOSSIBLE_WIN_RATE,
            severity=Severity.CRITICAL,
            title=f"Impossible win rate: {addr[:10]}...",
            description=(
                f"{profile.wins}/{profile.total_resolved} wins at avg entry "
                f"{profile.avg_entry_price:.2f}. "
                f"P-value: {result.win_rate_p_value:.2e}."
            ),
            wallet_address=addr,
            data={
                "wins": profile.wins,
                "total_resolved": profile.total_resolved,
                "p_value": str(result.win_rate_p_value),
                "pnl": str(profile.total_pnl),
            },
        )

    # SINGLE_MARKET_ALL_IN
    if (profile.total_volume >= config.single_market_min_volume
            and profile.markets_traded <= config.single_market_max_markets
            and profile.total_volume > ZERO
            and (profile.top_market_volume / profile.total_volume)
            >= config.single_market_min_pct):
        pct = profile.top_market_volume / profile.total_volume
        await emit(
            session,
            alert_type=AlertType.SINGLE_MARKET_ALL_IN,
            severity=Severity.HIGH,
            title=f"All-in concentration: {addr[:10]}...",
            description=(
                f"{pct * 100:.0f}% of {profile.total_volume:,.0f} USD "
                f"concentrated in a single market across "
                f"{profile.markets_traded} markets traded."
            ),
            wallet_address=addr,
            data={"top_market_pct": float(pct),
                  "total_volume": str(profile.total_volume)},
        )

    # FRESH_WHALE - wallet < 7 days old just placed a large trade.
    if profile.first_seen and profile.first_trade:
        wallet_age_days = (utc_now() - profile.first_seen).days
        if wallet_age_days <= config.fresh_whale_max_age_days:
            recent_large = await session.execute(
                select(Trade).where(
                    Trade.wallet_address == addr,
                    Trade.is_large.is_(True),
                    Trade.timestamp >= utc_now() - timedelta(hours=24),
                ).limit(1)
            )
            large_trade = recent_large.scalar_one_or_none()
            if (large_trade is not None
                    and Decimal(large_trade.size) >= config.fresh_whale_min_size_usd):
                await emit(
                    session,
                    alert_type=AlertType.FRESH_WHALE,
                    severity=Severity.MEDIUM,
                    title=f"Fresh whale: {addr[:10]}...",
                    description=(
                        f"Wallet ({wallet_age_days}d old) placed "
                        f"${Decimal(large_trade.size):,.0f} {large_trade.side} "
                        f"{large_trade.outcome} at {Decimal(large_trade.price):.2f}."
                    ),
                    wallet_address=addr,
                    market_id=large_trade.market_id,
                    data={"size": str(large_trade.size),
                          "age_days": wallet_age_days,
                          "price": str(large_trade.price)},
                )

    # RESOLUTION_SNIPE - trade placed < 6h pre-resolution at extreme odds.
    recent_snipe = await session.execute(
        select(Trade).where(
            Trade.wallet_address == addr,
            Trade.hours_before_resolution.is_not(None),
            Trade.hours_before_resolution < config.resolution_snipe_max_hours,
            Trade.price < config.resolution_snipe_max_price,
            Trade.timestamp >= utc_now() - timedelta(days=7),
        ).limit(1)
    )
    snipe = recent_snipe.scalar_one_or_none()
    if snipe is not None:
        await emit(
            session,
            alert_type=AlertType.RESOLUTION_SNIPE,
            severity=Severity.HIGH,
            title=f"Resolution snipe: {addr[:10]}...",
            description=(
                f"Bought at {Decimal(snipe.price):.2f}, "
                f"{Decimal(snipe.hours_before_resolution):.1f}h before "
                f"resolution. {'WON' if snipe.trade_won else 'OPEN/LOST'}."
            ),
            wallet_address=addr,
            market_id=snipe.market_id,
            data={"price": str(snipe.price),
                  "hours_before": str(snipe.hours_before_resolution)},
        )


async def score_wallet(
    address: str,
    *,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> InsiderScoreResult | None:
    """Score a single wallet, update its row, fire alerts, return the result."""
    async with db_session() as session:
        bundle = await _build_profile(session, address)
        if bundle is None:
            return None
        wallet, profile = bundle

        result = compute_insider_score(profile, config)
        smart = compute_smart_score(profile)
        prev_score = Decimal(wallet.insider_score or 0)
        win_rate = (Decimal(profile.wins) / Decimal(profile.total_resolved)
                    if profile.total_resolved else ZERO)

        await session.execute(
            update(Wallet)
            .where(Wallet.address == address)
            .values(
                insider_score=result.composite,
                smart_score=smart.composite,
                score_breakdown=result.breakdown.as_dict(),
                win_rate_p_value=result.win_rate_p_value,
                total_resolved=profile.total_resolved,
                wins=profile.wins,
                win_rate=win_rate,
                markets_traded=profile.markets_traded,
                total_trades=profile.total_trades,
                total_volume=profile.total_volume,
                total_pnl=profile.total_pnl,
                avg_entry_price=profile.avg_entry_price or None,
                avg_trade_size=profile.avg_trade_size or None,
                avg_hours_before_resolution=profile.avg_hours_before_resolution,
                top_market_volume=profile.top_market_volume,
                top_category_volume=profile.top_category_volume,
                political_military_volume=profile.political_military_volume,
                classification=result.classification.value,
            )
        )

        await _check_alerts(session, wallet, profile, result,
                            config=config, prev_score=prev_score)
        return result


async def score_recent(*, config: ScoringConfig = DEFAULT_CONFIG) -> int:
    """Score every wallet with activity in the last 2x scoring intervals."""
    window = timedelta(seconds=settings.scoring_interval * 2)
    cutoff = utc_now() - window
    async with db_session() as session:
        res = await session.execute(
            select(Wallet.address).where(Wallet.last_active >= cutoff)
        )
        addresses = [row[0] for row in res.all()]

    log.info("scoring_batch_start", count=len(addresses), cutoff=cutoff.isoformat())
    scored = 0
    for addr in addresses:
        try:
            if await score_wallet(addr, config=config):
                scored += 1
        except Exception:
            log.exception("scoring_wallet_error", wallet=addr)
    log.info("scoring_batch_done", scored=scored, total=len(addresses))
    return scored


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.scoring_interval
    log.info("scoring_loop_start", interval_sec=interval)
    while not stop_event.is_set():
        try:
            await score_recent()
        except Exception:
            log.exception("scoring_loop_error")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            pass
