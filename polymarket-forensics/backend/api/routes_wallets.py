"""
/wallets, /wallets/{address}, /wallets/{address}/trades, /wallets/{address}/funding
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import FundingChain, Trade, Wallet
from services import chain_tracer, scoring_engine

router = APIRouter(prefix="/wallets", tags=["wallets"])


def _serialize_wallet(w: Wallet) -> dict:
    return {
        "address": w.address,
        "first_seen": w.first_seen,
        "last_active": w.last_active,
        "total_trades": w.total_trades,
        "total_volume": float(w.total_volume or 0),
        "total_pnl": float(w.total_pnl or 0),
        "win_rate": float(w.win_rate or 0),
        "win_rate_p_value": float(w.win_rate_p_value or 1),
        "total_resolved": w.total_resolved,
        "wins": w.wins,
        "markets_traded": w.markets_traded,
        "avg_entry_price": float(w.avg_entry_price) if w.avg_entry_price else None,
        "avg_trade_size": float(w.avg_trade_size) if w.avg_trade_size else None,
        "avg_hours_before_resolution": float(w.avg_hours_before_resolution)
            if w.avg_hours_before_resolution else None,
        "top_market_volume": float(w.top_market_volume or 0),
        "top_category_volume": float(w.top_category_volume or 0),
        "political_military_volume": float(w.political_military_volume or 0),
        "insider_score": float(w.insider_score or 0),
        "smart_score": float(w.smart_score or 0),
        "score_breakdown": w.score_breakdown,
        "cluster_id": w.cluster_id,
        "classification": w.classification,
        "funding_source": w.funding_source,
        "funding_exchange": w.funding_exchange,
        "ens_name": w.ens_name,
        "notes": w.notes,
    }


@router.get("")
async def list_wallets(
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    classification: str | None = None,
    cluster_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("insider_score"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    q = select(Wallet).where(Wallet.insider_score >= min_score)
    if classification:
        q = q.where(Wallet.classification == classification)
    if cluster_id:
        q = q.where(Wallet.cluster_id == cluster_id)

    order_col = {
        "insider_score": Wallet.insider_score,
        "smart_score": Wallet.smart_score,
        "total_pnl": Wallet.total_pnl,
        "total_volume": Wallet.total_volume,
        "last_active": Wallet.last_active,
        "win_rate": Wallet.win_rate,
    }.get(order, Wallet.insider_score)

    total_q = await db.execute(select(func.count()).select_from(q.subquery()))
    total = total_q.scalar() or 0

    q = q.order_by(desc(order_col)).limit(limit).offset(offset)
    res = await db.execute(q)
    wallets = [_serialize_wallet(w) for w in res.scalars()]

    return {"total": total, "limit": limit, "offset": offset, "wallets": wallets}


@router.get("/{address}")
async def get_wallet(address: str, db: AsyncSession = Depends(get_db)) -> dict:
    addr = address.lower()
    res = await db.execute(select(Wallet).where(Wallet.address == addr))
    wallet = res.scalar_one_or_none()
    if wallet is None:
        raise HTTPException(404, "wallet not found")
    return _serialize_wallet(wallet)


@router.get("/{address}/trades")
async def get_wallet_trades(
    address: str,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    addr = address.lower()
    res = await db.execute(
        select(Trade).where(Trade.wallet_address == addr)
        .order_by(desc(Trade.timestamp)).limit(limit)
    )
    return [{
        "id": t.id,
        "market_id": t.market_id,
        "market_question": t.market_question,
        "market_category": t.market_category,
        "side": t.side,
        "outcome": t.outcome,
        "size": float(t.size),
        "price": float(t.price),
        "timestamp": t.timestamp,
        "tx_hash": t.tx_hash,
        "is_large": t.is_large,
        "hours_before_resolution": float(t.hours_before_resolution)
            if t.hours_before_resolution else None,
        "resolution_outcome": t.resolution_outcome,
        "trade_won": t.trade_won,
        "pnl": float(t.pnl) if t.pnl else None,
    } for t in res.scalars()]


@router.get("/{address}/funding")
async def get_wallet_funding(address: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    addr = address.lower()
    res = await db.execute(
        select(FundingChain).where(FundingChain.wallet_address == addr)
        .order_by(FundingChain.depth, desc(FundingChain.amount))
    )
    return [{
        "source_address": f.source_address,
        "source_type": f.source_type,
        "source_exchange": f.source_exchange,
        "amount": float(f.amount) if f.amount else None,
        "asset": f.asset,
        "timestamp": f.timestamp,
        "tx_hash": f.tx_hash,
        "depth": f.depth,
    } for f in res.scalars()]


@router.post("/{address}/trace")
async def request_trace(address: str) -> dict:
    addr = address.lower()
    await chain_tracer.enqueue(addr)
    return {"queued": True, "address": addr}


@router.post("/{address}/rescore")
async def rescore_wallet(address: str) -> dict:
    addr = address.lower()
    result = await scoring_engine.score_wallet(addr)
    if result is None:
        raise HTTPException(404, "wallet has no trades")
    return result


@router.patch("/{address}")
async def update_wallet_notes(
    address: str, payload: dict, db: AsyncSession = Depends(get_db),
) -> dict:
    addr = address.lower()
    res = await db.execute(select(Wallet).where(Wallet.address == addr))
    wallet = res.scalar_one_or_none()
    if wallet is None:
        raise HTTPException(404, "wallet not found")
    if "notes" in payload:
        wallet.notes = payload["notes"]
    if "classification" in payload:
        wallet.classification = payload["classification"]
    if "ens_name" in payload:
        wallet.ens_name = payload["ens_name"]
    await db.commit()
    return _serialize_wallet(wallet)
