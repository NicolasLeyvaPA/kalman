"""
Statistical tests used by the scoring engine.
"""
from __future__ import annotations

from scipy.stats import binom


def win_rate_p_value(wins: int, total: int, avg_implied_prob: float) -> float:
    """
    One-sided binomial test: probability of seeing >= `wins` successes in
    `total` trials when each trial's success probability is `avg_implied_prob`
    (the average implied probability the trader paid).

    A wallet that paid 18c on average and won 14/14 has p ~ 1e-12 —
    statistically impossible by luck. That's the signal we want.
    """
    if total <= 0 or wins <= 0:
        return 1.0
    p = max(min(avg_implied_prob, 0.999), 0.001)
    return float(1.0 - binom.cdf(wins - 1, total, p))


def volume_anomaly_z_score(
    current_volume: float,
    baseline_mean: float,
    baseline_std: float,
) -> float:
    if baseline_std <= 0:
        return 0.0
    return (current_volume - baseline_mean) / baseline_std
