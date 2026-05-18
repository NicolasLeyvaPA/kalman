"""Search routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import MarketResponse, SearchResponse, WalletResponse
from data.database import get_db
from data.models import Market, Wallet

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=2, max_length=128),
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    needle = f"%{q.lower()}%"

    wres = await db.execute(
        select(Wallet).where(
            or_(
                Wallet.address.ilike(needle),
                Wallet.ens_name.ilike(needle),
                Wallet.notes.ilike(needle),
            )
        ).limit(20)
    )
    mres = await db.execute(
        select(Market).where(
            or_(
                Market.id.ilike(needle),
                Market.question.ilike(needle),
            )
        ).limit(20)
    )

    return SearchResponse(
        wallets=[WalletResponse.model_validate(w) for w in wres.scalars()],
        markets=[MarketResponse.model_validate(m) for m in mres.scalars()],
    )
