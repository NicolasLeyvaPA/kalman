"""CSV / JSON export endpoints for offline analysis."""

from __future__ import annotations

import csv
import io
import json
from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import Alert, Trade, Wallet

router = APIRouter(prefix="/export", tags=["export"])


def _json_default(o: object) -> object:
    if isinstance(o, Decimal):
        return str(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o)}")


@router.get("/wallets.csv")
async def export_wallets_csv(
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(10000, ge=1, le=100000),
    db: AsyncSession = Depends(get_db),
) -> Response:
    res = await db.execute(
        select(Wallet).where(Wallet.insider_score >= min_score)
        .order_by(desc(Wallet.insider_score)).limit(limit)
    )
    wallets = list(res.scalars())

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "address", "classification", "insider_score", "smart_score",
        "win_rate", "win_rate_p_value", "total_trades", "total_resolved",
        "wins", "total_volume", "total_pnl", "markets_traded",
        "cluster_id", "funding_exchange", "first_seen", "last_active",
    ])
    for w in wallets:
        writer.writerow([
            w.address, w.classification, w.insider_score, w.smart_score,
            w.win_rate, w.win_rate_p_value, w.total_trades, w.total_resolved,
            w.wins, w.total_volume, w.total_pnl, w.markets_traded,
            w.cluster_id or "", w.funding_exchange or "",
            w.first_seen.isoformat() if w.first_seen else "",
            w.last_active.isoformat() if w.last_active else "",
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=wallets.csv"},
    )


@router.get("/trades.csv")
async def export_trades_csv(
    wallet: str | None = None,
    market: str | None = None,
    limit: int = Query(50000, ge=1, le=200000),
    db: AsyncSession = Depends(get_db),
) -> Response:
    q = select(Trade)
    if wallet:
        q = q.where(Trade.wallet_address == wallet.lower())
    if market:
        q = q.where(Trade.market_id == market)
    q = q.order_by(desc(Trade.timestamp)).limit(limit)
    res = await db.execute(q)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "wallet_address", "market_id", "market_category", "side",
        "outcome", "size", "price", "timestamp", "tx_hash", "is_large",
        "hours_before_resolution", "resolution_outcome", "trade_won", "pnl",
    ])
    for t in res.scalars():
        writer.writerow([
            t.id, t.wallet_address, t.market_id, t.market_category or "",
            t.side, t.outcome, t.size, t.price,
            t.timestamp.isoformat() if t.timestamp else "",
            t.tx_hash or "", t.is_large, t.hours_before_resolution or "",
            t.resolution_outcome or "", t.trade_won if t.trade_won is not None else "",
            t.pnl or "",
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trades.csv"},
    )


@router.get("/alerts.json")
async def export_alerts_json(
    limit: int = Query(1000, ge=1, le=10000),
    db: AsyncSession = Depends(get_db),
) -> Response:
    res = await db.execute(
        select(Alert).order_by(desc(Alert.created_at)).limit(limit)
    )
    payload = [
        {
            "id": a.id,
            "alert_type": a.alert_type,
            "severity": a.severity,
            "title": a.title,
            "description": a.description,
            "wallet_address": a.wallet_address,
            "cluster_id": a.cluster_id,
            "market_id": a.market_id,
            "data": a.data,
            "dismissed": a.dismissed,
            "created_at": a.created_at,
        }
        for a in res.scalars()
    ]
    body = json.dumps(payload, default=_json_default, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=alerts.json"},
    )
