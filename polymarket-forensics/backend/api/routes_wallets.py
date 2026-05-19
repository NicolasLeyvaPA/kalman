"""Wallet routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    FundingHopResponse,
    TraceResponse,
    TradeResponse,
    WalletListResponse,
    WalletPatch,
    WalletResponse,
)
from data.database import get_db
from enums import Classification
from repositories.wallets import WalletRepository
from services import chain_tracer, scoring_engine

router = APIRouter(prefix="/wallets", tags=["wallets"])


@router.get("", response_model=WalletListResponse)
async def list_wallets(
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    classification: str | None = None,
    cluster_id: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("insider_score"),
    db: AsyncSession = Depends(get_db),
) -> WalletListResponse:
    repo = WalletRepository(db)
    total, wallets = await repo.list_(
        min_score=min_score,
        classification=classification,
        cluster_id=cluster_id,
        order=order,
        limit=limit,
        offset=offset,
    )
    return WalletListResponse(
        total=total, limit=limit, offset=offset,
        wallets=[WalletResponse.model_validate(w) for w in wallets],
    )


@router.get("/{address}", response_model=WalletResponse)
async def get_wallet(address: str, db: AsyncSession = Depends(get_db)) -> WalletResponse:
    repo = WalletRepository(db)
    wallet = await repo.get(address)
    if wallet is None:
        raise HTTPException(404, "wallet not found")
    return WalletResponse.model_validate(wallet)


@router.get("/{address}/trades", response_model=list[TradeResponse])
async def get_wallet_trades(
    address: str,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> list[TradeResponse]:
    repo = WalletRepository(db)
    trades = await repo.list_trades(address, limit=limit)
    return [TradeResponse.model_validate(t) for t in trades]


@router.get("/{address}/funding", response_model=list[FundingHopResponse])
async def get_wallet_funding(
    address: str, db: AsyncSession = Depends(get_db),
) -> list[FundingHopResponse]:
    repo = WalletRepository(db)
    hops = await repo.list_funding(address)
    return [FundingHopResponse.model_validate(h) for h in hops]


@router.post("/{address}/trace", response_model=TraceResponse)
async def request_trace(address: str) -> TraceResponse:
    addr = address.lower()
    await chain_tracer.enqueue(addr)
    return TraceResponse(queued=True, address=addr)


@router.post("/{address}/rescore", response_model=WalletResponse)
async def rescore_wallet(
    address: str, db: AsyncSession = Depends(get_db),
) -> WalletResponse:
    addr = address.lower()
    try:
        result = await scoring_engine.score_wallet(addr)
    except Exception as exc:
        raise HTTPException(500, f"scoring failed: {exc}") from exc
    if result is None:
        raise HTTPException(404, "wallet has no trades")
    repo = WalletRepository(db)
    wallet = await repo.get(addr)
    if wallet is None:
        raise HTTPException(404, "wallet not found after scoring")
    return WalletResponse.model_validate(wallet)


@router.patch("/{address}", response_model=WalletResponse)
async def update_wallet(
    address: str, patch: WalletPatch, db: AsyncSession = Depends(get_db),
) -> WalletResponse:
    if patch.classification is not None:
        try:
            Classification(patch.classification)
        except ValueError as exc:
            raise HTTPException(400, f"invalid classification: {patch.classification}") from exc

    repo = WalletRepository(db)
    wallet = await repo.get(address)
    if wallet is None:
        raise HTTPException(404, "wallet not found")
    await repo.patch(
        wallet,
        notes=patch.notes,
        classification=patch.classification,
        ens_name=patch.ens_name,
    )
    await db.commit()
    return WalletResponse.model_validate(wallet)
