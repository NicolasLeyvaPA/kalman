from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import Cluster, Wallet

router = APIRouter(prefix="/clusters", tags=["clusters"])


def _serialize(c: Cluster) -> dict:
    return {
        "id": c.id,
        "wallets": c.wallets,
        "cluster_type": c.cluster_type,
        "evidence": c.evidence,
        "total_pnl": float(c.total_pnl or 0),
        "combined_win_rate": float(c.combined_win_rate or 0),
        "markets_in_common": c.markets_in_common or [],
        "insider_probability": float(c.insider_probability or 0),
        "first_detected": c.first_detected,
        "last_updated": c.last_updated,
        "status": c.status,
    }


@router.get("")
async def list_clusters(
    min_prob: float = 0.0, db: AsyncSession = Depends(get_db),
) -> list[dict]:
    res = await db.execute(
        select(Cluster).where(Cluster.insider_probability >= min_prob)
        .order_by(desc(Cluster.insider_probability))
    )
    return [_serialize(c) for c in res.scalars()]


@router.get("/{cluster_id}")
async def get_cluster(cluster_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    res = await db.execute(select(Cluster).where(Cluster.id == cluster_id))
    c = res.scalar_one_or_none()
    if c is None:
        raise HTTPException(404, "cluster not found")

    wres = await db.execute(select(Wallet).where(Wallet.address.in_(c.wallets)))
    wallets = [{
        "address": w.address,
        "insider_score": float(w.insider_score or 0),
        "total_pnl": float(w.total_pnl or 0),
        "total_volume": float(w.total_volume or 0),
        "classification": w.classification,
        "funding_exchange": w.funding_exchange,
    } for w in wres.scalars()]

    payload = _serialize(c)
    payload["wallet_details"] = wallets
    return payload


@router.get("/graph/edges")
async def get_graph_edges(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Returns nodes (wallets) + edges (cluster memberships) for the force-graph.
    """
    res = await db.execute(
        select(Wallet).where(Wallet.cluster_id.is_not(None))
        .order_by(desc(Wallet.insider_score)).limit(500)
    )
    wallets = list(res.scalars())

    nodes = [{
        "id": w.address,
        "label": w.ens_name or (w.address[:6] + "..." + w.address[-4:]),
        "insider_score": float(w.insider_score or 0),
        "total_volume": float(w.total_volume or 0),
        "cluster_id": w.cluster_id,
        "classification": w.classification,
    } for w in wallets]

    cres = await db.execute(select(Cluster))
    clusters = list(cres.scalars())
    edges: list[dict] = []
    for c in clusters:
        if len(c.wallets) < 2:
            continue
        anchor = c.wallets[0]
        for other in c.wallets[1:]:
            edges.append({
                "source": anchor,
                "target": other,
                "cluster_id": c.id,
                "type": c.cluster_type,
                "weight": float(c.insider_probability or 0),
            })

    return {"nodes": nodes, "edges": edges}
