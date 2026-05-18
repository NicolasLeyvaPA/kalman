"""
Volume / timing anomaly detection.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable


def detect_volume_spike(
    trades: Iterable[dict],
    market_id: str,
    window_hours: float = 4.0,
    baseline_hours: float = 168.0,
    spike_ratio: float = 5.0,
) -> dict | None:
    """
    Returns a payload if the recent window's volume exceeds `spike_ratio` x
    the trailing baseline mean. Otherwise None.
    """
    now = datetime.now().astimezone()
    window_start = now - timedelta(hours=window_hours)
    baseline_start = now - timedelta(hours=baseline_hours)

    recent_volume = 0.0
    baseline_volume = 0.0
    baseline_buckets: list[float] = []
    bucket_size = timedelta(hours=window_hours)
    buckets: dict[datetime, float] = defaultdict(float)

    for t in trades:
        if t.get("market_id") != market_id:
            continue
        ts = t.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except Exception:
                continue
        if ts is None or ts < baseline_start:
            continue

        size = float(t.get("size") or 0)
        if ts >= window_start:
            recent_volume += size
        else:
            baseline_volume += size
            bucket_key = baseline_start + bucket_size * int(
                (ts - baseline_start) / bucket_size
            )
            buckets[bucket_key] += size

    if not buckets:
        return None

    baseline_buckets = list(buckets.values())
    baseline_mean = sum(baseline_buckets) / len(baseline_buckets)
    if baseline_mean <= 0:
        return None

    ratio = recent_volume / baseline_mean
    if ratio < spike_ratio:
        return None

    return {
        "market_id": market_id,
        "recent_volume": recent_volume,
        "baseline_mean": baseline_mean,
        "ratio": ratio,
        "window_hours": window_hours,
    }
