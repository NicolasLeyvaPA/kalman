"""
Insider score: composite of 8 weighted sub-signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from scoring.statistical_tests import win_rate_p_value


WEIGHTS = {
    "freshness":         0.15,
    "concentration":     0.15,
    "win_anomaly":       0.20,
    "timing":            0.20,
    "size_odds":         0.10,
    "single_purpose":    0.08,
    "sensitive_markets": 0.07,
    "cluster":           0.05,
}


@dataclass
class WalletProfile:
    address: str
    first_seen: Optional[datetime]
    first_trade: Optional[datetime]
    total_trades: int
    total_volume: float
    total_pnl: float
    wins: int
    total_resolved: int
    avg_entry_price: float
    avg_trade_size: float
    avg_pnl_per_trade: float
    avg_hours_before_resolution: Optional[float]
    top_market_volume: float
    top_category_volume: float
    political_military_volume: float
    unique_protocols: int
    total_tx_count: int
    markets_traded: int
    cluster_id: Optional[str]
    cluster_insider_prob: float


def _freshness(p: WalletProfile) -> float:
    if not p.first_seen or not p.first_trade:
        return 0.0
    days = max(0.0, (p.first_trade - p.first_seen).total_seconds() / 86400.0)
    if days <= 1:
        return 1.0
    if days <= 7:
        return 0.7
    if days <= 30:
        return 0.3
    return 0.0


def _concentration(p: WalletProfile) -> float:
    if p.total_volume <= 0:
        return 0.0
    top_market = p.top_market_volume / p.total_volume
    top_category = p.top_category_volume / p.total_volume
    if top_market > 0.90:
        return 1.0
    if top_market > 0.70:
        return 0.7
    if top_category > 0.85:
        return 0.5
    return max(0.0, top_market - 0.3)


def _win_anomaly(p: WalletProfile) -> tuple[float, float]:
    if p.total_resolved < 5:
        return 0.0, 1.0
    pv = win_rate_p_value(p.wins, p.total_resolved, p.avg_entry_price)
    if pv < 0.0001:
        return 1.0, pv
    if pv < 0.001:
        return 0.85, pv
    if pv < 0.01:
        return 0.6, pv
    if pv < 0.05:
        return 0.3, pv
    return 0.0, pv


def _timing(p: WalletProfile) -> float:
    h = p.avg_hours_before_resolution
    if h is None:
        return 0.0
    if h < 6:
        return 1.0
    if h < 24:
        return 0.7
    if h < 72:
        return 0.3
    return 0.0


def _size_odds(p: WalletProfile) -> float:
    if p.avg_trade_size <= 0:
        return 0.0
    edge = p.avg_pnl_per_trade / p.avg_trade_size
    if edge > 3.0:
        return 1.0
    if edge > 1.5:
        return 0.7
    if edge > 0.5:
        return 0.3
    return 0.0


def _single_purpose(p: WalletProfile) -> float:
    if p.unique_protocols <= 2 and p.total_tx_count < 20:
        return 1.0
    if p.unique_protocols <= 3:
        return 0.5
    return 0.0


def _sensitive_markets(p: WalletProfile) -> float:
    if p.total_volume <= 0:
        return 0.0
    pct = p.political_military_volume / p.total_volume
    if pct > 0.80:
        return 1.0
    if pct > 0.50:
        return 0.6
    return 0.0


def _cluster(p: WalletProfile) -> float:
    if not p.cluster_id:
        return 0.0
    if p.cluster_insider_prob > 0.7:
        return 1.0
    if p.cluster_insider_prob > 0.4:
        return 0.5
    return 0.0


def compute_insider_score(p: WalletProfile) -> dict[str, float | dict]:
    """
    Returns:
      {
        "insider_score": 0.0-1.0,
        "win_rate_p_value": float,
        "breakdown": { signal: 0.0-1.0, ... },
      }
    """
    win_anom, pv = _win_anomaly(p)
    breakdown = {
        "freshness":         round(_freshness(p), 4),
        "concentration":     round(_concentration(p), 4),
        "win_anomaly":       round(win_anom, 4),
        "timing":            round(_timing(p), 4),
        "size_odds":         round(_size_odds(p), 4),
        "single_purpose":    round(_single_purpose(p), 4),
        "sensitive_markets": round(_sensitive_markets(p), 4),
        "cluster":           round(_cluster(p), 4),
    }
    score = sum(breakdown[k] * WEIGHTS[k] for k in WEIGHTS)
    return {
        "insider_score": round(score, 4),
        "win_rate_p_value": pv,
        "breakdown": breakdown,
    }


def classification_from_score(score: float) -> str:
    if score >= 0.7:
        return "insider_suspect"
    if score >= 0.5:
        return "suspicious"
    if score >= 0.3:
        return "watch"
    return "normal"
