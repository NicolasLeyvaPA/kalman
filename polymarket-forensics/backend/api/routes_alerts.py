"""Alert routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import AlertListResponse, AlertResponse, DismissResponse
from data.database import get_db
from repositories.alerts import AlertRepository

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    severity: str | None = None,
    dismissed: bool = False,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> AlertListResponse:
    repo = AlertRepository(db)
    total, alerts = await repo.list_(
        severity=severity, dismissed=dismissed, limit=limit, offset=offset,
    )
    return AlertListResponse(
        total=total, limit=limit, offset=offset,
        alerts=[AlertResponse.model_validate(a) for a in alerts],
    )


@router.post("/{alert_id}/dismiss", response_model=DismissResponse)
async def dismiss_alert(
    alert_id: int, db: AsyncSession = Depends(get_db),
) -> DismissResponse:
    repo = AlertRepository(db)
    ok = await repo.dismiss(alert_id)
    if not ok:
        raise HTTPException(404, "alert not found")
    await db.commit()
    return DismissResponse(dismissed=True, id=alert_id)
