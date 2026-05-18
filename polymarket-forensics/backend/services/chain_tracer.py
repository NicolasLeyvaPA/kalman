"""
On-demand chain-tracer service. A coroutine queue: wallets are enqueued
when they cross the insider trace threshold or when the operator requests
a manual trace from the UI.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from decimal import Decimal

from sqlalchemy import select, delete, update

from chain.alchemy_client import get_alchemy
from chain.funding_tracer import trace_funding_chain
from config import get_settings
from data.database import db_session
from data.models import FundingChain, Wallet
from utils.logging import get_logger

log = get_logger("tracer")
settings = get_settings()

_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
_seen_recent: set[str] = set()


async def enqueue(address: str) -> None:
    addr = address.lower()
    if addr in _seen_recent:
        return
    _seen_recent.add(addr)
    if len(_seen_recent) > 5000:
        _seen_recent.clear()
    try:
        _queue.put_nowait(addr)
    except asyncio.QueueFull:
        log.warning("trace queue full, dropping %s", addr)


async def _trace_and_store(address: str) -> int:
    alchemy = await get_alchemy()
    if not alchemy.api_key:
        log.warning("ALCHEMY_API_KEY not set, skipping trace")
        return 0
    try:
        hops = await trace_funding_chain(address, alchemy, max_depth=3, max_per_hop=25)
    except Exception as exc:
        log.warning("trace failed for %s: %s", address, exc)
        return 0

    if not hops:
        return 0

    async with db_session() as session:
        await session.execute(
            delete(FundingChain).where(FundingChain.wallet_address == address)
        )
        primary_exchange = None
        primary_source = None
        for h in hops:
            session.add(FundingChain(
                wallet_address=h.wallet_address,
                source_address=h.source_address,
                source_type=h.source_type,
                source_exchange=h.source_exchange,
                amount=Decimal(str(h.amount)) if h.amount else None,
                asset=h.asset,
                timestamp=h.timestamp,
                tx_hash=h.tx_hash,
                depth=h.depth,
            ))
            if h.depth == 0 and h.source_type == "exchange" and primary_exchange is None:
                primary_exchange = h.source_exchange
                primary_source = h.source_address

        await session.execute(
            update(Wallet).where(Wallet.address == address).values(
                funding_source=primary_source,
                funding_exchange=primary_exchange,
            )
        )

    return len(hops)


async def run_worker(stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            address = await asyncio.wait_for(_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            continue
        n = await _trace_and_store(address)
        log.info("traced %s: %d hops", address, n)
        _queue.task_done()


async def enqueue_high_score_wallets() -> int:
    async with db_session() as session:
        res = await session.execute(
            select(Wallet.address).where(
                (Wallet.insider_score >= settings.insider_trace_threshold)
                & (Wallet.funding_source.is_(None))
            ).limit(50)
        )
        addresses = [row[0] for row in res.all()]
    for a in addresses:
        await enqueue(a)
    return len(addresses)
