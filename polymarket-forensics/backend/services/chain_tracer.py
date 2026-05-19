"""On-demand chain-tracing worker.

Wallets are enqueued when they cross the insider trace threshold, when an
operator clicks "Trace" in the UI, or when a large trade arrives from a
brand-new wallet. The worker pops one address at a time, walks the funding
chain, and persists every hop.
"""

from __future__ import annotations

import asyncio

from sqlalchemy import delete, select, update

from chain.alchemy_client import get_alchemy
from chain.funding_tracer import trace_funding_chain
from config import get_settings
from data.database import db_session
from data.models import FundingChain, Wallet
from enums import SourceType
from exceptions import AlchemyConfigError, ExternalAPIError
from utils.logging import get_logger

log = get_logger(__name__)
settings = get_settings()

_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
_recent_ttl: dict[str, float] = {}
_RECENT_TTL_SEC = 3600
_TRACE_NEEDS_CLUSTERS = asyncio.Event()


def _prune_recent() -> None:
    cutoff = asyncio.get_event_loop().time() - _RECENT_TTL_SEC
    stale = [k for k, ts in _recent_ttl.items() if ts < cutoff]
    for k in stale:
        _recent_ttl.pop(k, None)


async def enqueue(address: str) -> None:
    """Add a wallet to the trace queue. No-op if traced recently."""
    addr = address.lower()
    loop = asyncio.get_event_loop()
    _prune_recent()
    if addr in _recent_ttl:
        return
    _recent_ttl[addr] = loop.time()
    try:
        _queue.put_nowait(addr)
    except asyncio.QueueFull:
        log.warning("trace_queue_full", dropped=addr)


def needs_clustering() -> bool:
    """Reset and return whether new traces have been added since last clear."""
    if _TRACE_NEEDS_CLUSTERS.is_set():
        _TRACE_NEEDS_CLUSTERS.clear()
        return True
    return False


async def _trace_and_store(address: str) -> int:
    try:
        alchemy = await get_alchemy()
    except Exception:
        log.exception("trace_alchemy_init_failed", wallet=address)
        return 0
    if not alchemy.api_key:
        log.warning("trace_skipped_no_key", wallet=address)
        return 0

    try:
        hops = await trace_funding_chain(address, alchemy, max_depth=3, max_per_hop=25)
    except AlchemyConfigError as exc:
        log.warning("trace_config_error", wallet=address, error=exc.message)
        return 0
    except ExternalAPIError as exc:
        log.warning("trace_external_error",
                    wallet=address, status=exc.status, service=exc.service)
        return 0
    except Exception:
        log.exception("trace_unexpected_error", wallet=address)
        return 0

    if not hops:
        log.info("trace_empty", wallet=address)
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
                source_type=h.source_type.value,
                source_exchange=h.source_exchange,
                amount=h.amount if h.amount else None,
                asset=h.asset,
                timestamp=h.timestamp,
                tx_hash=h.tx_hash,
                depth=h.depth,
            ))
            if (h.depth == 0
                    and h.source_type == SourceType.EXCHANGE
                    and primary_exchange is None):
                primary_exchange = h.source_exchange
                primary_source = h.source_address

        await session.execute(
            update(Wallet).where(Wallet.address == address).values(
                funding_source=primary_source,
                funding_exchange=primary_exchange,
            )
        )

    _TRACE_NEEDS_CLUSTERS.set()
    log.info("trace_complete", wallet=address, hops=len(hops),
             exchange=primary_exchange)
    return len(hops)


async def run_worker(stop_event: asyncio.Event) -> None:
    log.info("trace_worker_start")
    while not stop_event.is_set():
        try:
            address = await asyncio.wait_for(_queue.get(), timeout=2.0)
        except TimeoutError:
            continue
        try:
            await _trace_and_store(address)
        finally:
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
