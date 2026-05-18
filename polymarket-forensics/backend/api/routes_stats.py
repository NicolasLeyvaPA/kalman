from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import Alert, Cluster, Trade, Wallet

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/overview")
async def overview(db: AsyncSession = Depends(get_db)) -> dict:
    wcount = (await db.execute(select(func.count()).select_from(Wallet))).scalar() or 0
    tcount = (await db.execute(select(func.count()).select_from(Trade))).scalar() or 0
    ccount = (await db.execute(select(func.count()).select_from(Cluster))).scalar() or 0

    insider = (await db.execute(
        select(func.count()).select_from(Wallet).where(Wallet.insider_score >= 0.7)
    )).scalar() or 0

    suspect = (await db.execute(
        select(func.count()).select_from(Wallet).where(
            (Wallet.insider_score >= 0.5) & (Wallet.insider_score < 0.7)
        )
    )).scalar() or 0

    open_alerts = (await db.execute(
        select(func.count()).select_from(Alert).where(Alert.dismissed.is_(False))
    )).scalar() or 0

    critical = (await db.execute(
        select(func.count()).select_from(Alert).where(
            Alert.dismissed.is_(False), Alert.severity == "critical"
        )
    )).scalar() or 0

    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=24)
    vol24h_q = await db.execute(
        select(func.sum(Trade.size)).where(Trade.timestamp >= cutoff)
    )
    vol24h = float(vol24h_q.scalar() or 0)

    large24h_q = await db.execute(
        select(func.count()).select_from(Trade).where(
            (Trade.timestamp >= cutoff) & (Trade.is_large.is_(True))
        )
    )
    large24h = large24h_q.scalar() or 0

    return {
        "wallets_total": wcount,
        "trades_total": tcount,
        "clusters_total": ccount,
        "insider_suspects": insider,
        "suspicious": suspect,
        "open_alerts": open_alerts,
        "critical_alerts": critical,
        "volume_24h": vol24h,
        "large_trades_24h": large24h,
    }


@router.get("/recent-trades")
async def recent_trades(limit: int = 30, db: AsyncSession = Depends(get_db)) -> list[dict]:
    res = await db.execute(
        select(Trade, Wallet)
        .join(Wallet, Wallet.address == Trade.wallet_address)
        .order_by(desc(Trade.timestamp)).limit(limit)
    )
    out = []
    for trade, wallet in res.all():
        out.append({
            "id": trade.id,
            "wallet_address": trade.wallet_address,
            "wallet_ens": wallet.ens_name,
            "wallet_insider_score": float(wallet.insider_score or 0),
            "wallet_classification": wallet.classification,
            "market_id": trade.market_id,
            "market_question": trade.market_question,
            "market_category": trade.market_category,
            "side": trade.side,
            "outcome": trade.outcome,
            "size": float(trade.size),
            "price": float(trade.price),
            "timestamp": trade.timestamp,
            "is_large": trade.is_large,
        })
    return out
