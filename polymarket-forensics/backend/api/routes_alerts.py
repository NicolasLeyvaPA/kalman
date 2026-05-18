from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import Alert

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("")
async def list_alerts(
    severity: str | None = None,
    dismissed: bool = False,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    q = select(Alert).where(Alert.dismissed == dismissed)
    if severity:
        q = q.where(Alert.severity == severity)
    q = q.order_by(desc(Alert.created_at)).limit(limit)
    res = await db.execute(q)
    return [{
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
    } for a in res.scalars()]


@router.post("/{alert_id}/dismiss")
async def dismiss_alert(alert_id: int, db: AsyncSession = Depends(get_db)) -> dict:
    res = await db.execute(select(Alert).where(Alert.id == alert_id))
    alert = res.scalar_one_or_none()
    if alert is None:
        raise HTTPException(404, "alert not found")
    await db.execute(update(Alert).where(Alert.id == alert_id).values(dismissed=True))
    await db.commit()
    return {"dismissed": True, "id": alert_id}
