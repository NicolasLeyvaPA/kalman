"""Alert repository."""

from __future__ import annotations

from sqlalchemy import desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import Alert


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_(
        self,
        *,
        severity: str | None = None,
        dismissed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[Alert]]:
        q = select(Alert).where(Alert.dismissed == dismissed)
        if severity:
            q = q.where(Alert.severity == severity)

        total_res = await self._session.execute(
            select(func.count()).select_from(q.subquery())
        )
        total = int(total_res.scalar() or 0)

        page_q = q.order_by(desc(Alert.created_at)).limit(limit).offset(offset)
        res = await self._session.execute(page_q)
        return total, list(res.scalars())

    async def dismiss(self, alert_id: int) -> bool:
        res = await self._session.execute(
            update(Alert).where(Alert.id == alert_id).values(dismissed=True)
            .returning(Alert.id)
        )
        return res.scalar_one_or_none() is not None
