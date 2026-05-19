"""Typed result objects for the scoring engine.

Scoring functions are pure: they take a ``WalletProfile`` snapshot plus a
``ScoringConfig`` and return a typed result. No side effects, no I/O,
no database. This makes them trivially testable and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from enums import Classification


@dataclass(frozen=True)
class WalletProfile:
    """Immutable snapshot of a wallet's aggregate state at scoring time."""

    address: str
    first_seen: datetime | None
    first_trade: datetime | None
    total_trades: int
    total_volume: Decimal
    total_pnl: Decimal
    wins: int
    total_resolved: int
    avg_entry_price: Decimal
    avg_trade_size: Decimal
    avg_pnl_per_trade: Decimal
    avg_hours_before_resolution: Decimal | None
    top_market_volume: Decimal
    top_category_volume: Decimal
    political_military_volume: Decimal
    unique_protocols: int
    total_tx_count: int
    markets_traded: int
    cluster_id: str | None
    cluster_insider_prob: Decimal


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-signal contributions to the composite insider score."""

    freshness: Decimal
    concentration: Decimal
    win_anomaly: Decimal
    timing: Decimal
    size_odds: Decimal
    single_purpose: Decimal
    sensitive_markets: Decimal
    cluster: Decimal

    def as_dict(self) -> dict[str, float]:
        return {
            "freshness":         float(self.freshness),
            "concentration":     float(self.concentration),
            "win_anomaly":       float(self.win_anomaly),
            "timing":            float(self.timing),
            "size_odds":         float(self.size_odds),
            "single_purpose":    float(self.single_purpose),
            "sensitive_markets": float(self.sensitive_markets),
            "cluster":           float(self.cluster),
        }


@dataclass(frozen=True)
class InsiderScoreResult:
    """Output of the insider-score computation."""

    composite: Decimal
    win_rate_p_value: Decimal
    breakdown: ScoreBreakdown
    classification: Classification


@dataclass(frozen=True)
class SmartScoreResult:
    composite: Decimal


@dataclass(frozen=True)
class ScoringConfig:
    """All thresholds and weights live here. Tuning happens by editing this
    class or by overriding fields from a YAML/env-driven loader.

    Weights MUST sum to 1.0 (validated post-init).
    """

    # --- weights ---------------------------------------------------------
    w_freshness:         Decimal = Decimal("0.15")
    w_concentration:     Decimal = Decimal("0.15")
    w_win_anomaly:       Decimal = Decimal("0.20")
    w_timing:            Decimal = Decimal("0.20")
    w_size_odds:         Decimal = Decimal("0.10")
    w_single_purpose:    Decimal = Decimal("0.08")
    w_sensitive_markets: Decimal = Decimal("0.07")
    w_cluster:           Decimal = Decimal("0.05")

    # --- freshness thresholds (days between first_seen and first_trade) -
    fresh_critical_days: int = 1
    fresh_high_days:     int = 7
    fresh_low_days:      int = 30

    # --- concentration (share of volume in single market / category) ----
    concentration_critical: Decimal = Decimal("0.90")
    concentration_high:     Decimal = Decimal("0.70")
    category_high:          Decimal = Decimal("0.85")

    # --- win-rate p-value thresholds ------------------------------------
    p_critical: Decimal = Decimal("0.0001")
    p_high:     Decimal = Decimal("0.001")
    p_medium:   Decimal = Decimal("0.01")
    p_low:      Decimal = Decimal("0.05")
    win_anomaly_min_resolved: int = 5

    # --- timing (hours before resolution) -------------------------------
    timing_critical_hours: Decimal = Decimal("6")
    timing_high_hours:     Decimal = Decimal("24")
    timing_low_hours:      Decimal = Decimal("72")

    # --- size/odds (pnl/size edge multiple) -----------------------------
    edge_critical: Decimal = Decimal("3.0")
    edge_high:     Decimal = Decimal("1.5")
    edge_low:      Decimal = Decimal("0.5")

    # --- single-purpose wallet ------------------------------------------
    single_purpose_max_protocols: int = 2
    single_purpose_max_tx:        int = 20

    # --- sensitive markets ----------------------------------------------
    sensitive_critical: Decimal = Decimal("0.80")
    sensitive_high:     Decimal = Decimal("0.50")

    # --- cluster --------------------------------------------------------
    cluster_critical_prob: Decimal = Decimal("0.7")
    cluster_high_prob:     Decimal = Decimal("0.4")

    # --- classification thresholds --------------------------------------
    class_insider_suspect: Decimal = Decimal("0.70")
    class_suspicious:      Decimal = Decimal("0.50")
    class_watch:           Decimal = Decimal("0.30")

    # --- alert generators ------------------------------------------------
    fresh_whale_max_age_days: int = 7
    fresh_whale_min_size_usd: Decimal = Decimal("10000")
    single_market_min_volume: Decimal = Decimal("5000")
    single_market_min_pct:    Decimal = Decimal("0.90")
    single_market_max_markets: int = 2
    resolution_snipe_max_hours: Decimal = Decimal("6")
    resolution_snipe_max_price: Decimal = Decimal("0.20")
    impossible_winrate_min_resolved: int = 10

    def __post_init__(self) -> None:
        total = (
            self.w_freshness + self.w_concentration + self.w_win_anomaly
            + self.w_timing + self.w_size_odds + self.w_single_purpose
            + self.w_sensitive_markets + self.w_cluster
        )
        if abs(total - Decimal("1.0")) > Decimal("0.0001"):
            raise ValueError(f"scoring weights must sum to 1.0, got {total}")


DEFAULT_CONFIG = ScoringConfig()
