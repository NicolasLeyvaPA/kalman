"""
Cluster detection: groups wallets that are likely the same entity.

Three methods:
  1. Funding-linked  — share an immediate funding source
  2. Temporal        — bet the same side on the same market within 2h
  3. Behavioral      — cosine similarity on profile feature vectors
"""
from __future__ import annotations

import asyncio
import hashlib
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Iterable

from sqlalchemy import select, update

from api.websocket import manager
from config import get_settings
from data.database import db_session
from data.models import Alert, Cluster, FundingChain, Trade, Wallet
from utils.logging import get_logger

log = get_logger("cluster")
settings = get_settings()


def _cluster_id(prefix: str, members: Iterable[str]) -> str:
    key = ",".join(sorted(set(members)))
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"C-{prefix}-{h}"


async def detect_funding_clusters() -> list[dict]:
    async with db_session() as session:
        res = await session.execute(select(FundingChain))
        chains = list(res.scalars())

    groups: dict[tuple[str, str | None], set[str]] = defaultdict(set)
    for c in chains:
        if c.source_type not in ("exchange", "wallet"):
            continue
        key = (c.source_address.lower(), c.source_exchange)
        groups[key].add(c.wallet_address.lower())

    clusters = []
    for (src, exch), wallets in groups.items():
        if len(wallets) < 2:
            continue
        clusters.append({
            "wallets": sorted(wallets),
            "type": "funding_linked",
            "evidence": f"Funded from {exch or src[:10] + '...'}",
        })
    return clusters


async def detect_temporal_clusters(window_hours: float = 2.0) -> list[dict]:
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=7)
    async with db_session() as session:
        res = await session.execute(
            select(Trade).where(Trade.timestamp >= cutoff)
            .order_by(Trade.market_id, Trade.timestamp)
        )
        trades = list(res.scalars())

    by_market: dict[str, list[Trade]] = defaultdict(list)
    for t in trades:
        by_market[t.market_id].append(t)

    clusters = []
    window = timedelta(hours=window_hours)
    seen_groups: set[tuple[str, ...]] = set()
    for market_id, mtrades in by_market.items():
        mtrades.sort(key=lambda t: t.timestamp)
        for i, anchor in enumerate(mtrades):
            window_addresses: dict[str, datetime] = {anchor.wallet_address: anchor.timestamp}
            for t in mtrades[i + 1:]:
                if t.timestamp - anchor.timestamp > window:
                    break
                if t.side == anchor.side and t.outcome == anchor.outcome:
                    window_addresses.setdefault(t.wallet_address, t.timestamp)
            if len(window_addresses) >= 3:
                key = tuple(sorted(window_addresses.keys()))
                if key in seen_groups:
                    continue
                seen_groups.add(key)
                clusters.append({
                    "wallets": list(key),
                    "type": "temporal",
                    "evidence": f"{len(key)} wallets, same side on market {market_id} within {window_hours}h",
                    "market_id": market_id,
                })
    return clusters


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
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def detect_behavioral_clusters(min_sim: float = 0.92,
                                       min_score: float = 0.4) -> list[dict]:
    async with db_session() as session:
        res = await session.execute(
            select(Wallet).where(Wallet.insider_score >= min_score)
        )
        wallets = list(res.scalars())

    if len(wallets) < 2:
        return []

    vecs = [(w.address, _wallet_feature_vector(w)) for w in wallets]
    parent = {a: a for a, _ in vecs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(vecs)):
        ai, vi = vecs[i]
        for j in range(i + 1, len(vecs)):
            aj, vj = vecs[j]
            if _cosine(vi, vj) >= min_sim:
                union(ai, aj)

    groups: dict[str, list[str]] = defaultdict(list)
    for a, _ in vecs:
        groups[find(a)].append(a)

    clusters = []
    for members in groups.values():
        if len(members) >= 2:
            clusters.append({
                "wallets": sorted(members),
                "type": "behavioral",
                "evidence": f"Behavioral similarity > {min_sim:.2f}",
            })
    return clusters


def _merge_overlapping(clusters: list[dict]) -> list[dict]:
    if not clusters:
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

    for c in clusters:
        members = c["wallets"]
        for m in members:
            find(m)
        for m in members[1:]:
            union(members[0], m)

    groups: dict[str, dict] = {}
    for c in clusters:
        root = find(c["wallets"][0])
        g = groups.setdefault(root, {
            "wallets": set(),
            "types": set(),
            "evidence": [],
            "markets": set(),
        })
        g["wallets"].update(c["wallets"])
        g["types"].add(c["type"])
        g["evidence"].append(c["evidence"])
        if c.get("market_id"):
            g["markets"].add(c["market_id"])

    merged: list[dict] = []
    for g in groups.values():
        merged.append({
            "wallets": sorted(g["wallets"]),
            "type": "_".join(sorted(g["types"])),
            "evidence": " · ".join(g["evidence"][:3]),
            "markets_in_common": sorted(g["markets"]),
        })
    return merged


async def _compute_cluster_insider_prob(session, members: list[str]) -> float:
    res = await session.execute(
        select(Wallet.insider_score).where(Wallet.address.in_(members))
    )
    scores = [float(s or 0) for (s,) in res.all()]
    if not scores:
        return 0.0
    avg = sum(scores) / len(scores)
    boost = min(0.15, 0.03 * max(0, len(members) - 2))
    return min(1.0, avg + boost)


async def persist_clusters(clusters: list[dict]) -> int:
    n = 0
    async with db_session() as session:
        for c in clusters:
            members = c["wallets"]
            cid = _cluster_id(c["type"].split("_")[0][:3], members)

            insider_prob = await _compute_cluster_insider_prob(session, members)

            existing = await session.execute(select(Cluster).where(Cluster.id == cid))
            row = existing.scalar_one_or_none()
            if row is None:
                cluster = Cluster(
                    id=cid,
                    wallets=members,
                    cluster_type=c["type"],
                    evidence=c["evidence"],
                    markets_in_common=c.get("markets_in_common") or [],
                    insider_probability=insider_prob,
                )
                session.add(cluster)
                n += 1
                if insider_prob >= 0.5 and "funding" in c["type"]:
                    alert = Alert(
                        alert_type="FUNDING_CHAIN_MATCH",
                        severity="critical",
                        title=f"Funding cluster: {len(members)} wallets",
                        description=f"{c['evidence']}. Combined insider probability {insider_prob:.2f}.",
                        cluster_id=cid,
                        data={"wallets": members, "evidence": c["evidence"]},
                    )
                    session.add(alert)
                    await session.flush()
                    await manager.broadcast({
                        "type": "alert",
                        "id": alert.id,
                        "alert_type": "FUNDING_CHAIN_MATCH",
                        "severity": "critical",
                        "title": alert.title,
                        "description": alert.description,
                        "cluster_id": cid,
                    })
                elif "temporal" in c["type"] and len(members) >= 3:
                    alert = Alert(
                        alert_type="PRE_EVENT_CLUSTER",
                        severity="high",
                        title=f"Temporal cluster: {len(members)} wallets",
                        description=c["evidence"],
                        cluster_id=cid,
                        data={"wallets": members, "evidence": c["evidence"]},
                    )
                    session.add(alert)
                    await session.flush()
                    await manager.broadcast({
                        "type": "alert",
                        "id": alert.id,
                        "alert_type": "PRE_EVENT_CLUSTER",
                        "severity": "high",
                        "title": alert.title,
                        "description": alert.description,
                        "cluster_id": cid,
                    })
            else:
                row.wallets = members
                row.evidence = c["evidence"]
                row.markets_in_common = c.get("markets_in_common") or row.markets_in_common
                row.insider_probability = insider_prob
                row.last_updated = datetime.now(tz=timezone.utc)

            for m in members:
                await session.execute(
                    update(Wallet).where(Wallet.address == m).values(cluster_id=cid)
                )

    return n


async def run_once() -> dict[str, int]:
    funding = await detect_funding_clusters()
    temporal = await detect_temporal_clusters()
    behavioral = await detect_behavioral_clusters()
    merged = _merge_overlapping(funding + temporal + behavioral)
    new_clusters = await persist_clusters(merged)
    return {
        "funding": len(funding),
        "temporal": len(temporal),
        "behavioral": len(behavioral),
        "merged": len(merged),
        "new": new_clusters,
    }


async def run_loop(stop_event: asyncio.Event) -> None:
    interval = settings.cluster_interval
    while not stop_event.is_set():
        try:
            stats = await run_once()
            log.info("clusters: %s", stats)
        except Exception as exc:
            log.exception("cluster loop error: %s", exc)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
