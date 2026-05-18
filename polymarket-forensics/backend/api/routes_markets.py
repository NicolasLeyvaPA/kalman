from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import Market, Trade, Wallet

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("")
async def list_markets(
    category: str | None = None,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    q = select(Market)
    if category:
        q = q.where(Market.category == category)
    if status:
        q = q.where(Market.status == status)
    q = q.order_by(desc(Market.volume_total)).limit(limit)
    res = await db.execute(q)
    return [{
        "id": m.id,
        "question": m.question,
        "category": m.category,
        "status": m.status,
        "resolution_outcome": m.resolution_outcome,
        "current_price": float(m.current_price or 0),
        "volume_total": float(m.volume_total or 0),
        "resolved_at": m.resolved_at,
    } for m in res.scalars()]


@router.get("/{market_id}/forensics")
async def market_forensics(market_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    mres = await db.execute(select(Market).where(Market.id == market_id))
    market = mres.scalar_one_or_none()
    if market is None:
        raise HTTPException(404, "market not found")

    sub = (
        select(
            Trade.wallet_address,
            func.sum(Trade.size).label("vol"),
            func.count().label("trades"),
        ).where(Trade.market_id == market_id).group_by(Trade.wallet_address)
    ).subquery()

    q = (
        select(
            Wallet.address, Wallet.insider_score, Wallet.classification,
            Wallet.funding_exchange, sub.c.vol, sub.c.trades,
        )
        .join(sub, Wallet.address == sub.c.wallet_address)
        .order_by(desc(Wallet.insider_score))
        .limit(100)
    )
    res = await db.execute(q)
    rows = res.all()

    dirty_vol = sum(float(r.vol or 0) for r in rows if float(r.insider_score or 0) >= 0.5)
    clean_vol = sum(float(r.vol or 0) for r in rows if float(r.insider_score or 0) < 0.5)

    return {
        "market": {
            "id": market.id,
            "question": market.question,
            "category": market.category,
            "status": market.status,
            "resolution_outcome": market.resolution_outcome,
            "volume_total": float(market.volume_total or 0),
        },
        "dirty_volume": dirty_vol,
        "clean_volume": clean_vol,
        "wallets": [{
            "address": r.address,
            "insider_score": float(r.insider_score or 0),
            "classification": r.classification,
            "funding_exchange": r.funding_exchange,
            "volume": float(r.vol or 0),
            "trades": int(r.trades or 0),
        } for r in rows],
    }


@router.get("/{market_id}/trades")
async def market_trades(
    market_id: str,
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    res = await db.execute(
        select(Trade).where(Trade.market_id == market_id)
        .order_by(desc(Trade.timestamp)).limit(limit)
    )
    return [{
        "id": t.id,
        "wallet_address": t.wallet_address,
        "side": t.side,
        "outcome": t.outcome,
        "size": float(t.size),
        "price": float(t.price),
        "timestamp": t.timestamp,
        "is_large": t.is_large,
    } for t in res.scalars()]
