"""Centralized alert emission.

Every alert in the system goes through ``emit()`` here. That makes it
trivial to swap the broadcast channel, dedupe identical alerts within a
short window, or apply severity-based suppression policies.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.websocket import manager
from data.models import Alert
from enums import AlertType, Severity
from utils.logging import get_logger
from utils.time import utc_now

log = get_logger(__name__)


# Suppress duplicate alerts of the same type for the same target within this window.
DEDUP_WINDOW = timedelta(hours=6)


async def _recently_emitted(
    session: AsyncSession,
    *,
    alert_type: AlertType,
    wallet_address: str | None = None,
    cluster_id: str | None = None,
    market_id: str | None = None,
) -> bool:
    cutoff = utc_now() - DEDUP_WINDOW
    stmt = select(Alert.id).where(
        Alert.alert_type == alert_type.value,
        Alert.created_at >= cutoff,
    )
    if wallet_address is not None:
        stmt = stmt.where(Alert.wallet_address == wallet_address)
    if cluster_id is not None:
        stmt = stmt.where(Alert.cluster_id == cluster_id)
    if market_id is not None:
        stmt = stmt.where(Alert.market_id == market_id)
    res = await session.execute(stmt.limit(1))
    return res.scalar_one_or_none() is not None


async def emit(
    session: AsyncSession,
    *,
    alert_type: AlertType,
    severity: Severity,
    title: str,
    description: str,
    wallet_address: str | None = None,
    cluster_id: str | None = None,
    market_id: str | None = None,
    data: dict[str, Any] | None = None,
    dedupe: bool = True,
) -> Alert | None:
    """Persist an alert and broadcast it. Returns the Alert on success, or
    None if the alert was deduped.

    The caller's outer ``db_session`` context manager is responsible for
    committing the transaction.
    """
    if dedupe:
        suppressed = await _recently_emitted(
            session,
            alert_type=alert_type,
            wallet_address=wallet_address,
            cluster_id=cluster_id,
            market_id=market_id,
        )
        if suppressed:
            log.debug("alert_deduped",
                      type=alert_type.value, wallet=wallet_address,
                      cluster=cluster_id, market=market_id)
            return None

    alert = Alert(
        alert_type=alert_type.value,
        severity=severity.value,
        title=title,
        description=description,
        wallet_address=wallet_address,
        cluster_id=cluster_id,
        market_id=market_id,
        data=data or {},
    )
    session.add(alert)
    await session.flush()

    log.info("alert_emitted",
             id=alert.id, type=alert_type.value, severity=severity.value,
             wallet=wallet_address, cluster=cluster_id, market=market_id)

    await manager.broadcast({
        "type": "alert",
        "id": alert.id,
        "alert_type": alert_type.value,
        "severity": severity.value,
        "title": title,
        "description": description,
        "wallet_address": wallet_address,
        "cluster_id": cluster_id,
        "market_id": market_id,
        "data": data or {},
        "created_at": utc_now().isoformat(),
    })

    return alert
