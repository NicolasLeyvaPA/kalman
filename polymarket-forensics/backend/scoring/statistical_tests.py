"""Statistical primitives used by the scoring engine."""

from __future__ import annotations

from decimal import Decimal

from scipy.stats import binom

from exceptions import InsufficientDataError

EPSILON = Decimal("0.001")
MAX_PROB = Decimal("0.999")


def win_rate_p_value(wins: int, total: int, avg_implied_prob: Decimal) -> Decimal:
    """One-sided binomial p-value.

    P(X >= wins | n=total, p=avg_implied_prob)

    A wallet that paid 18c on average and won 14/14 has p ~ 1e-12 — that is
    statistically impossible by luck and is the signal we want to catch.

    Args:
        wins:             Number of resolved trades the wallet won.
        total:            Number of resolved trades.
        avg_implied_prob: Average price paid (≈ market-implied probability).

    Returns:
        p-value in (0, 1]. Returns 1 if there are no trades or no wins.
    """
    if total <= 0 or wins <= 0:
        return Decimal("1.0")
    if wins > total:
        raise ValueError(f"wins ({wins}) cannot exceed total ({total})")
    p = float(min(max(avg_implied_prob, EPSILON), MAX_PROB))
    raw = float(1.0 - binom.cdf(wins - 1, total, p))
    return Decimal(str(max(raw, 1e-18)))


def volume_anomaly_z_score(
    current_volume: Decimal,
    baseline_mean: Decimal,
    baseline_std: Decimal,
) -> Decimal:
    """Z-score of current_volume against the trailing baseline distribution.

    Raises:
        InsufficientDataError: if baseline_std is non-positive.
    """
    if baseline_std <= 0:
        raise InsufficientDataError(
            "baseline std must be positive to compute z-score",
            context={"baseline_std": str(baseline_std)},
        )
    return (current_volume - baseline_mean) / baseline_std
