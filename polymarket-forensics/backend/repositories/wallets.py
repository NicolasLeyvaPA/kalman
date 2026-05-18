"""Wallet repository — all SQL touching the wallets/trades/funding tables.

Route handlers and services use this; nothing else writes SQL against
these tables directly.
"""

from __future__ import annotations

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import FundingChain, Trade, Wallet


class WalletRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, address: str) -> Wallet | None:
        res = await self._session.execute(
            select(Wallet).where(Wallet.address == address.lower())
        )
        return res.scalar_one_or_none()

    async def list_(
        self,
        *,
        min_score: float = 0.0,
        classification: str | None = None,
        cluster_id: str | None = None,
        order: str = "insider_score",
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[int, list[Wallet]]:
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

        total_res = await self._session.execute(
            select(func.count()).select_from(q.subquery())
        )
        total = int(total_res.scalar() or 0)

        page_q = q.order_by(desc(order_col)).limit(limit).offset(offset)
        res = await self._session.execute(page_q)
        return total, list(res.scalars())

    async def list_trades(
        self, address: str, *, limit: int = 100,
    ) -> list[Trade]:
        res = await self._session.execute(
            select(Trade).where(Trade.wallet_address == address.lower())
            .order_by(desc(Trade.timestamp)).limit(limit)
        )
        return list(res.scalars())

    async def list_funding(self, address: str) -> list[FundingChain]:
        res = await self._session.execute(
            select(FundingChain).where(
                FundingChain.wallet_address == address.lower()
            ).order_by(FundingChain.depth, desc(FundingChain.amount))
        )
        return list(res.scalars())

    async def patch(self, wallet: Wallet, *, notes: str | None = None,
                    classification: str | None = None,
                    ens_name: str | None = None) -> Wallet:
        if notes is not None:
            wallet.notes = notes
        if classification is not None:
            wallet.classification = classification
        if ens_name is not None:
            wallet.ens_name = ens_name
        await self._session.flush()
        return wallet
