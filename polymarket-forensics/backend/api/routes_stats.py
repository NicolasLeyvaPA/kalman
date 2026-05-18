"""Dashboard summary statistics + scheduler status."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    RecentTradeResponse,
    SchedulerStatusResponse,
    ServiceStatusResponse,
    StatsResponse,
)
from data.database import get_db
from data.models import Alert, Cluster, Trade, Wallet
from services.scheduler import scheduler
from utils.time import utc_now

router = APIRouter(prefix="/stats", tags=["stats"])


async def _count(db: AsyncSession, stmt) -> int:
    res = await db.execute(stmt)
    return int(res.scalar() or 0)


@router.get("/overview", response_model=StatsResponse)
async def overview(db: AsyncSession = Depends(get_db)) -> StatsResponse:
    wcount = await _count(db, select(func.count()).select_from(Wallet))
    tcount = await _count(db, select(func.count()).select_from(Trade))
    ccount = await _count(db, select(func.count()).select_from(Cluster))

    insider = await _count(
        db, select(func.count()).select_from(Wallet)
        .where(Wallet.insider_score >= 0.7),
    )
    suspect = await _count(
        db, select(func.count()).select_from(Wallet)
        .where((Wallet.insider_score >= 0.5) & (Wallet.insider_score < 0.7)),
    )
    open_alerts = await _count(
        db, select(func.count()).select_from(Alert).where(Alert.dismissed.is_(False)),
    )
    critical = await _count(
        db, select(func.count()).select_from(Alert)
        .where(Alert.dismissed.is_(False), Alert.severity == "critical"),
    )

    cutoff = utc_now() - timedelta(hours=24)
    vol24 = await db.execute(
        select(func.sum(Trade.size)).where(Trade.timestamp >= cutoff)
    )
    large24 = await _count(
        db, select(func.count()).select_from(Trade)
        .where((Trade.timestamp >= cutoff) & (Trade.is_large.is_(True))),
    )

    return StatsResponse(
        wallets_total=wcount,
        trades_total=tcount,
        clusters_total=ccount,
        insider_suspects=insider,
        suspicious=suspect,
        open_alerts=open_alerts,
        critical_alerts=critical,
        volume_24h=vol24.scalar() or 0,
        large_trades_24h=large24,
    )


@router.get("/recent-trades", response_model=list[RecentTradeResponse])
async def recent_trades(
    limit: int = 30, db: AsyncSession = Depends(get_db),
) -> list[RecentTradeResponse]:
    res = await db.execute(
        select(Trade, Wallet)
        .join(Wallet, Wallet.address == Trade.wallet_address)
        .order_by(desc(Trade.timestamp)).limit(limit)
    )
    return [
        RecentTradeResponse(
            id=trade.id,
            wallet_address=trade.wallet_address,
            wallet_ens=wallet.ens_name,
            wallet_insider_score=wallet.insider_score or 0,
            wallet_classification=wallet.classification,
            market_id=trade.market_id,
            market_question=trade.market_question,
            market_category=trade.market_category,
            side=trade.side,
            outcome=trade.outcome,
            size=trade.size,
            price=trade.price,
            timestamp=trade.timestamp,
            is_large=trade.is_large,
        )
        for trade, wallet in res.all()
    ]


@router.get("/scheduler", response_model=SchedulerStatusResponse)
async def scheduler_status() -> SchedulerStatusResponse:
    return SchedulerStatusResponse(
        services=[
            ServiceStatusResponse(
                name=s.name, running=s.running, last_started=s.last_started,
            )
            for s in scheduler.status()
        ]
    )
