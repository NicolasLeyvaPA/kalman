from fastapi import APIRouter, Depends, Query
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.database import get_db
from data.models import Market, Wallet

router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search(
    q: str = Query(..., min_length=2),
    db: AsyncSession = Depends(get_db),
) -> dict:
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
    wallets = [{
        "address": w.address,
        "ens_name": w.ens_name,
        "insider_score": float(w.insider_score or 0),
        "classification": w.classification,
    } for w in wres.scalars()]

    mres = await db.execute(
        select(Market).where(
            or_(
                Market.id.ilike(needle),
                Market.question.ilike(needle),
            )
        ).limit(20)
    )
    markets = [{
        "id": m.id,
        "question": m.question,
        "category": m.category,
        "status": m.status,
    } for m in mres.scalars()]

    return {"wallets": wallets, "markets": markets}
