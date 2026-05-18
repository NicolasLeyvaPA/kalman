"""Market routes."""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    MarketForensicsResponse,
    MarketForensicsWallet,
    MarketResponse,
    TradeResponse,
)
from data.database import get_db
from data.models import Market, Trade, Wallet

router = APIRouter(prefix="/markets", tags=["markets"])


@router.get("", response_model=list[MarketResponse])
async def list_markets(
    category: str | None = None,
    status: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[MarketResponse]:
    q = select(Market)
    if category:
        q = q.where(Market.category == category)
    if status:
        q = q.where(Market.status == status)
    q = q.order_by(desc(Market.volume_total)).limit(limit)
    res = await db.execute(q)
    return [MarketResponse.model_validate(m) for m in res.scalars()]


@router.get("/{market_id}/forensics", response_model=MarketForensicsResponse)
async def market_forensics(
    market_id: str, db: AsyncSession = Depends(get_db),
) -> MarketForensicsResponse:
    mres = await db.execute(select(Market).where(Market.id == market_id))
    market = mres.scalar_one_or_none()
    if market is None:
        raise HTTPException(404, "market not found")

    sub = (
        select(
            Trade.wallet_address,
            func.sum(Trade.size).label("vol"),
            func.count().label("trades"),
        )
        .where(Trade.market_id == market_id)
        .group_by(Trade.wallet_address)
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

    dirty_vol = sum(
        (Decimal(r.vol or 0) for r in rows if Decimal(r.insider_score or 0) >= Decimal("0.5")),
        Decimal(0),
    )
    clean_vol = sum(
        (Decimal(r.vol or 0) for r in rows if Decimal(r.insider_score or 0) < Decimal("0.5")),
        Decimal(0),
    )

    return MarketForensicsResponse(
        market=MarketResponse.model_validate(market),
        dirty_volume=dirty_vol,
        clean_volume=clean_vol,
        wallets=[
            MarketForensicsWallet(
                address=r.address,
                insider_score=r.insider_score or Decimal(0),
                classification=r.classification,
                funding_exchange=r.funding_exchange,
                volume=r.vol or Decimal(0),
                trades=int(r.trades or 0),
            )
            for r in rows
        ],
    )


@router.get("/{market_id}/trades", response_model=list[TradeResponse])
async def market_trades(
    market_id: str,
    limit: int = Query(200, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
) -> list[TradeResponse]:
    res = await db.execute(
        select(Trade).where(Trade.market_id == market_id)
        .order_by(desc(Trade.timestamp)).limit(limit)
    )
    return [TradeResponse.model_validate(t) for t in res.scalars()]
