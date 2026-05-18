"""Cluster detection.

Three orthogonal methods, merged via union-find on member overlap:

  1. funding_linked — wallets sharing an immediate funding source.
  2. temporal       — wallets betting the same side, same market, within 2h.
  3. behavioral     — wallets with cosine-similar profile feature vectors.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from sqlalchemy import select, update

from config import get_settings
from data.database import db_session
from data.models import Cluster, FundingChain, Trade, Wallet
from enums import AlertType, ClusterType, Severity
from services.alert_generator import emit
from utils.logging import get_logger
from utils.time import utc_now

log = get_logger(__name__)
settings = get_settings()

TEMPORAL_WINDOW = timedelta(hours=2)
TEMPORAL_LOOKBACK = timedelta(days=7)
BEHAVIORAL_MIN_SIMILARITY = Decimal("0.92")
BEHAVIORAL_MIN_SCORE = Decimal("0.4")


@dataclass(frozen=True)
class CandidateCluster:
    wallets: tuple[str, ...]
    cluster_type: ClusterType
    evidence: str
    market_id: str | None = None


def _cluster_id(prefix: str, members: Iterable[str]) -> str:
    """Deterministic ID from a sorted member set."""
    key = ",".join(sorted({m.lower() for m in members}))
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"C-{prefix}-{h}"


async def detect_funding_clusters() -> list[CandidateCluster]:
    async with db_session() as session:
        res = await session.execute(select(FundingChain))
        chains = list(res.scalars())

    groups: dict[tuple[str, str | None], set[str]] = defaultdict(set)
    for c in chains:
        if c.source_type not in ("exchange", "wallet"):
            continue
        key = (c.source_address.lower(), c.source_exchange)
        groups[key].add(c.wallet_address.lower())

    out: list[CandidateCluster] = []
    for (src, exch), wallets in groups.items():
        if len(wallets) < 2:
            continue
        out.append(CandidateCluster(
            wallets=tuple(sorted(wallets)),
            cluster_type=ClusterType.FUNDING_LINKED,
            evidence=(
                f"Funded from {exch} ({src[:10]}...)"
                if exch else f"Funded from {src[:10]}..."
            ),
        ))
    return out


async def detect_temporal_clusters() -> list[CandidateCluster]:
    cutoff = utc_now() - TEMPORAL_LOOKBACK
    async with db_session() as session:
        res = await session.execute(
            select(Trade).where(Trade.timestamp >= cutoff)
            .order_by(Trade.market_id, Trade.timestamp)
        )
        trades = list(res.scalars())

    by_market: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_market[t.market_id].append(t)

    out: list[CandidateCluster] = []
    seen: set[tuple[str, ...]] = set()
    for market_id, mtrades in by_market.items():
        mtrades.sort(key=lambda t: t.timestamp)
        for i, anchor in enumerate(mtrades):
            members: dict[str, Trade] = {anchor.wallet_address: anchor}
            for t in mtrades[i + 1:]:
                if t.timestamp - anchor.timestamp > TEMPORAL_WINDOW:
                    break
                if t.side == anchor.side and t.outcome == anchor.outcome:
                    members.setdefault(t.wallet_address, t)
            if len(members) >= 3:
                key = tuple(sorted(members.keys()))
                if key in seen:
                    continue
                seen.add(key)
                out.append(CandidateCluster(
                    wallets=key,
                    cluster_type=ClusterType.TEMPORAL,
                    evidence=(
                        f"{len(key)} wallets betting {anchor.side} {anchor.outcome} "
                        f"within {TEMPORAL_WINDOW.total_seconds() / 3600:.0f}h"
                    ),
                    market_id=market_id,
                ))
    return out


def _wallet_feature_vector(w: Wallet) -> list[float]:
    total_vol = float(w.total_volume or 0) or 1.0
    return [
        float(w.win_rate or 0),
        float(w.avg_entry_price or 0),
        math.log10(1 + float(w.avg_trade_size or 0)),
        float(w.political_military_volume or 0) / total_vol,
        float(w.top_market_volume or 0) / total_vol,
        float(w.markets_traded or 0) / 50.0,
        float(w.insider_score or 0),
    ]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def detect_behavioral_clusters() -> list[CandidateCluster]:
    async with db_session() as session:
        res = await session.execute(
            select(Wallet).where(Wallet.insider_score >= BEHAVIORAL_MIN_SCORE)
        )
        wallets = list(res.scalars())

    if len(wallets) < 2:
        return []

    vecs = [(w.address, _wallet_feature_vector(w)) for w in wallets]
    parent: dict[str, str] = {a: a for a, _ in vecs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    threshold = float(BEHAVIORAL_MIN_SIMILARITY)
    for i, (ai, vi) in enumerate(vecs):
        for aj, vj in vecs[i + 1:]:
            if _cosine(vi, vj) >= threshold:
                union(ai, aj)

    groups: dict[str, list[str]] = defaultdict(list)
    for a, _ in vecs:
        groups[find(a)].append(a)

    out: list[CandidateCluster] = []
    for members in groups.values():
        if len(members) >= 2:
            out.append(CandidateCluster(
                wallets=tuple(sorted(members)),
                cluster_type=ClusterType.BEHAVIORAL,
                evidence=f"Behavioral similarity ≥ {BEHAVIORAL_MIN_SIMILARITY:.2f}",
            ))
    return out


def _merge_overlapping(candidates: list[CandidateCluster]) -> list[CandidateCluster]:
    """Union-find merge across all candidates that share at least one member."""
    if not candidates:
        return []

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for c in candidates:
        for m in c.wallets:
            find(m)
        anchor = c.wallets[0]
        for m in c.wallets[1:]:
            union(anchor, m)

    bucket: dict[str, dict[str, object]] = {}
    for c in candidates:
        root = find(c.wallets[0])
        b = bucket.setdefault(root, {
            "wallets": set(),
            "types": set(),
            "evidence": [],
            "markets": set(),
        })
        b["wallets"].update(c.wallets)
        b["types"].add(c.cluster_type)
        b["evidence"].append(c.evidence)
        if c.market_id:
            b["markets"].add(c.market_id)

    merged: list[CandidateCluster] = []
    for b in bucket.values():
        types_set: set[ClusterType] = b["types"]
        ctype = ClusterType.MIXED if len(types_set) > 1 else next(iter(types_set))
        merged.append(CandidateCluster(
            wallets=tuple(sorted(b["wallets"])),
            cluster_type=ctype,
            evidence=" · ".join(b["evidence"][:3]),
        ))
    return merged


async def _compute_cluster_prob(session, members: list[str]) -> Decimal:
    res = await session.execute(
        select(Wallet.insider_score).where(Wallet.address.in_(members))
    )
    scores = [Decimal(s or 0) for (s,) in res.all()]
    if not scores:
        return Decimal("0")
    avg = sum(scores, Decimal("0")) / Decimal(len(scores))
    boost = min(Decimal("0.15"), Decimal("0.03") * Decimal(max(0, len(members) - 2)))
    return min(Decimal("1.0"), avg + boost)


async def persist_clusters(candidates: list[CandidateCluster]) -> int:
    """Persist or update each cluster, fire one-time alerts on creation."""
    new_count = 0
    async with db_session() as session:
        for cand in candidates:
            members = list(cand.wallets)
            type_prefix = cand.cluster_type.value.split("_")[0][:3]
            cid = _cluster_id(type_prefix, members)

            insider_prob = await _compute_cluster_prob(session, members)

            existing = await session.execute(select(Cluster).where(Cluster.id == cid))
            row = existing.scalar_one_or_none()

            if row is None:
                session.add(Cluster(
                    id=cid,
                    wallets=members,
                    cluster_type=cand.cluster_type.value,
                    evidence=cand.evidence,
                    markets_in_common=[cand.market_id] if cand.market_id else [],
                    insider_probability=insider_prob,
                ))
                new_count += 1
                if (cand.cluster_type == ClusterType.FUNDING_LINKED
                        and insider_prob >= Decimal("0.5")):
                    await emit(
                        session,
                        alert_type=AlertType.FUNDING_CHAIN_MATCH,
                        severity=Severity.CRITICAL,
                        title=f"Funding cluster: {len(members)} wallets",
                        description=(
                            f"{cand.evidence}. Combined insider probability "
                            f"{insider_prob:.2f}."
                        ),
                        cluster_id=cid,
                        data={"wallets": members, "evidence": cand.evidence,
                              "insider_probability": str(insider_prob)},
                    )
                elif cand.cluster_type == ClusterType.TEMPORAL and len(members) >= 3:
                    await emit(
                        session,
                        alert_type=AlertType.PRE_EVENT_CLUSTER,
                        severity=Severity.HIGH,
                        title=f"Temporal cluster: {len(members)} wallets",
                        description=cand.evidence,
                        cluster_id=cid,
                        market_id=cand.market_id,
                        data={"wallets": members, "evidence": cand.evidence},
                    )
            else:
                row.wallets = members
                row.evidence = cand.evidence
                row.insider_probability = insider_prob
                row.last_updated = utc_now()

            for m in members:
                await session.execute(
                    update(Wallet).where(Wallet.address == m).values(cluster_id=cid)
                )
    return new_count


async def run_once() -> dict[str, int]:
    funding = await detect_funding_clusters()
    temporal = await detect_temporal_clusters()
    behavioral = await detect_behavioral_clusters()
    merged = _merge_overlapping(funding + temporal + behavioral)
    new_clusters = await persist_clusters(merged)
    stats = {
        "funding": len(funding),
        "temporal": len(temporal),
        "behavioral": len(behavioral),
        "merged": len(merged),
        "new": new_clusters,
    }
    log.info("clusters_detected", **stats)
    return stats


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.cluster_interval
    log.info("cluster_loop_start", interval_sec=interval)
    while not stop_event.is_set():
        try:
            await run_once()
        except Exception:
            log.exception("cluster_loop_error")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            pass
