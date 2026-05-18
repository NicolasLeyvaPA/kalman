"""Cluster routes."""

from __future__ import annotations

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    ClusterDetailResponse,
    ClusterGraphEdge,
    ClusterGraphNode,
    ClusterGraphResponse,
    ClusterResponse,
    ClusterWalletSummary,
)
from data.database import get_db
from data.models import Cluster, Wallet

router = APIRouter(prefix="/clusters", tags=["clusters"])


@router.get("", response_model=list[ClusterResponse])
async def list_clusters(
    min_prob: float = 0.0, db: AsyncSession = Depends(get_db),
) -> list[ClusterResponse]:
    res = await db.execute(
        select(Cluster).where(Cluster.insider_probability >= min_prob)
        .order_by(desc(Cluster.insider_probability))
    )
    return [ClusterResponse.model_validate(c) for c in res.scalars()]


@router.get("/{cluster_id}", response_model=ClusterDetailResponse)
async def get_cluster(
    cluster_id: str, db: AsyncSession = Depends(get_db),
) -> ClusterDetailResponse:
    res = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    c = res.scalar_one_or_none()
    if c is None:
        raise HTTPException(404, "cluster not found")

    wres = await db.execute(select(Wallet).where(Wallet.address.in_(c.wallets)))
    wallet_details = [
        ClusterWalletSummary(
            address=w.address,
            insider_score=w.insider_score or Decimal(0),
            total_pnl=w.total_pnl or Decimal(0),
            total_volume=w.total_volume or Decimal(0),
            classification=w.classification,
            funding_exchange=w.funding_exchange,
        )
        for w in wres.scalars()
    ]
    base = ClusterResponse.model_validate(c)
    return ClusterDetailResponse(**base.model_dump(), wallet_details=wallet_details)


@router.get("/graph/edges", response_model=ClusterGraphResponse)
async def get_graph(db: AsyncSession = Depends(get_db)) -> ClusterGraphResponse:
    res = await db.execute(
        select(Wallet).where(Wallet.cluster_id.is_not(None))
        .order_by(desc(Wallet.insider_score)).limit(500)
    )
    wallets = list(res.scalars())

    nodes = [
        ClusterGraphNode(
            id=w.address,
            label=w.ens_name or (w.address[:6] + "..." + w.address[-4:]),
            insider_score=w.insider_score or Decimal(0),
            total_volume=w.total_volume or Decimal(0),
            cluster_id=w.cluster_id,
            classification=w.classification,
        )
        for w in wallets
    ]

    cres = await db.execute(select(Cluster))
    edges: list[ClusterGraphEdge] = []
    for c in cres.scalars():
        if len(c.wallets) < 2:
            continue
        anchor = c.wallets[0]
        for other in c.wallets[1:]:
            edges.append(ClusterGraphEdge(
                source=anchor, target=other, cluster_id=c.id,
                type=c.cluster_type,
                weight=c.insider_probability or Decimal(0),
            ))

    return ClusterGraphResponse(nodes=nodes, edges=edges)
