"""Smart-money score — the inverse profile of an insider.

A smart trader looks like:
  - long operating history
  - diversified across many markets
  - positive win rate, but not absurdly so
  - positive risk-adjusted ROI

An insider looks like:
  - short history
  - concentrated in one market / category
  - statistically impossible win rate
  - extreme PnL relative to size

Both signals run on the same ``WalletProfile`` so a trader can in principle
have both a high smart-score and a high insider-score; in practice they're
strongly anti-correlated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

from scoring.types import SmartScoreResult, WalletProfile


@dataclass(frozen=True)
class SmartScoreConfig:
    high_resolved_threshold: int = 50
    mid_resolved_threshold:  int = 20
    high_markets_threshold:  int = 30
    mid_markets_threshold:   int = 10
    high_winrate:            Decimal = Decimal("0.65")
    mid_winrate:             Decimal = Decimal("0.55")
    long_history_days:       int = 90
    roi_signal_cap:          Decimal = Decimal("0.20")


DEFAULT_SMART_CONFIG = SmartScoreConfig()


def compute_smart_score(
    profile: WalletProfile,
    config: SmartScoreConfig = DEFAULT_SMART_CONFIG,
) -> SmartScoreResult:
    """Composite smart-money score in [0, 1]."""
    score = Decimal("0.0")

    if profile.total_resolved >= config.high_resolved_threshold:
        score += Decimal("0.25")
    elif profile.total_resolved >= config.mid_resolved_threshold:
        score += Decimal("0.15")

    if profile.markets_traded >= config.high_markets_threshold:
        score += Decimal("0.20")
    elif profile.markets_traded >= config.mid_markets_threshold:
        score += Decimal("0.10")

    if profile.total_resolved > 0:
        wr = Decimal(profile.wins) / Decimal(profile.total_resolved)
        if wr > config.high_winrate:
            score += Decimal("0.20")
        elif wr > config.mid_winrate:
            score += Decimal("0.10")

    if profile.total_pnl > 0 and profile.total_volume > 0:
        roi = profile.total_pnl / profile.total_volume
        roi_signal = Decimal(str(math.log10(1 + float(roi) * 10))) * Decimal("0.20")
        score += min(config.roi_signal_cap, max(Decimal("0.0"), roi_signal))

    if profile.first_seen and profile.first_trade:
        days = (profile.first_trade - profile.first_seen).total_seconds() / 86400.0
        if days > config.long_history_days:
            score += Decimal("0.15")

    return SmartScoreResult(
        composite=min(Decimal("1.0"), score).quantize(Decimal("0.0001")),
    )
