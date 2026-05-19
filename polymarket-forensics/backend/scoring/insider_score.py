"""Insider score: composite of eight weighted sub-signals.

This module is intentionally pure. All inputs come from a ``WalletProfile``
and a ``ScoringConfig``; the only output is an ``InsiderScoreResult``. No
database access, no logging, no network — every branch is unit-testable
in microseconds.
"""

from __future__ import annotations

from decimal import Decimal

from enums import Classification
from scoring.statistical_tests import win_rate_p_value
from scoring.types import (
    DEFAULT_CONFIG,
    InsiderScoreResult,
    ScoreBreakdown,
    ScoringConfig,
    WalletProfile,
)

ONE = Decimal("1.0")
ZERO = Decimal("0.0")


def _freshness(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """Score how new the wallet is at the moment of its first trade.

    A wallet created today that immediately places a large bet is
    indistinguishable from a wallet created specifically for that bet.
    """
    if p.first_seen is None or p.first_trade is None:
        return ZERO
    days = max(ZERO, Decimal((p.first_trade - p.first_seen).total_seconds()) / Decimal("86400"))
    if days <= cfg.fresh_critical_days:
        return ONE
    if days <= cfg.fresh_high_days:
        return Decimal("0.7")
    if days <= cfg.fresh_low_days:
        return Decimal("0.3")
    return ZERO


def _concentration(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """Share of volume concentrated in a single market or category."""
    if p.total_volume <= ZERO:
        return ZERO
    top_market_pct = p.top_market_volume / p.total_volume
    top_category_pct = p.top_category_volume / p.total_volume
    if top_market_pct > cfg.concentration_critical:
        return ONE
    if top_market_pct > cfg.concentration_high:
        return Decimal("0.7")
    if top_category_pct > cfg.category_high:
        return Decimal("0.5")
    excess = top_market_pct - Decimal("0.3")
    return max(ZERO, excess)


def _win_anomaly(p: WalletProfile, cfg: ScoringConfig) -> tuple[Decimal, Decimal]:
    """Binomial p-value of the observed wins given the average entry price.

    Returns (signal_score, raw_p_value). When there aren't enough resolved
    trades for the test to be meaningful, returns (0, 1) so the signal
    contributes nothing and the breakdown still shows the p-value.
    """
    if p.total_resolved < cfg.win_anomaly_min_resolved:
        return ZERO, ONE
    pv = win_rate_p_value(p.wins, p.total_resolved, p.avg_entry_price)
    if pv < cfg.p_critical:
        return ONE, pv
    if pv < cfg.p_high:
        return Decimal("0.85"), pv
    if pv < cfg.p_medium:
        return Decimal("0.6"), pv
    if pv < cfg.p_low:
        return Decimal("0.3"), pv
    return ZERO, pv


def _timing(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """How close to market resolution were the wallet's trades placed?"""
    h: Decimal | None = p.avg_hours_before_resolution
    if h is None:
        return ZERO
    if h < cfg.timing_critical_hours:
        return ONE
    if h < cfg.timing_high_hours:
        return Decimal("0.7")
    if h < cfg.timing_low_hours:
        return Decimal("0.3")
    return ZERO


def _size_odds(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """Edge multiple: PnL per trade divided by avg trade size.

    Betting $50K at 10c (10% implied probability) requires extreme
    confidence. People with public information don't size like that.
    """
    if p.avg_trade_size <= ZERO:
        return ZERO
    edge = p.avg_pnl_per_trade / p.avg_trade_size
    if edge > cfg.edge_critical:
        return ONE
    if edge > cfg.edge_high:
        return Decimal("0.7")
    if edge > cfg.edge_low:
        return Decimal("0.3")
    return ZERO


def _single_purpose(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """Wallet that only touches CEX + Polymarket and nothing else.

    Normal wallets have NFTs, DeFi, multiple dApps. Insiders set up a
    burner that only does the one thing they care about.
    """
    if (p.unique_protocols <= cfg.single_purpose_max_protocols
            and p.total_tx_count < cfg.single_purpose_max_tx):
        return ONE
    if p.unique_protocols <= cfg.single_purpose_max_protocols + 1:
        return Decimal("0.5")
    return ZERO


def _sensitive_markets(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """Share of volume in politically/militarily sensitive markets.

    These are the highest-signal insider markets — "will the US strike
    Iran" cannot be informed by public analysis the way "BTC > 150K" can.
    """
    if p.total_volume <= ZERO:
        return ZERO
    pct = p.political_military_volume / p.total_volume
    if pct > cfg.sensitive_critical:
        return ONE
    if pct > cfg.sensitive_high:
        return Decimal("0.6")
    return ZERO


def _cluster(p: WalletProfile, cfg: ScoringConfig) -> Decimal:
    """Is the wallet linked to other already-suspicious wallets?"""
    if not p.cluster_id:
        return ZERO
    if p.cluster_insider_prob > cfg.cluster_critical_prob:
        return ONE
    if p.cluster_insider_prob > cfg.cluster_high_prob:
        return Decimal("0.5")
    return ZERO


def classification_from_score(
    score: Decimal, cfg: ScoringConfig = DEFAULT_CONFIG,
) -> Classification:
    """Bucket a composite score into a coarse classification."""
    if score >= cfg.class_insider_suspect:
        return Classification.INSIDER_SUSPECT
    if score >= cfg.class_suspicious:
        return Classification.SUSPICIOUS
    if score >= cfg.class_watch:
        return Classification.WATCH
    return Classification.NORMAL


def compute_insider_score(
    profile: WalletProfile,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> InsiderScoreResult:
    """Compute the composite insider score and per-signal breakdown.

    Args:
        profile: Immutable wallet snapshot at scoring time.
        config:  Thresholds and weights. Defaults to ``DEFAULT_CONFIG``.

    Returns:
        ``InsiderScoreResult`` with composite score in [0, 1], the raw
        binomial p-value, the per-signal breakdown, and the derived
        classification.
    """
    win_anom_score, pv = _win_anomaly(profile, config)
    breakdown = ScoreBreakdown(
        freshness=_freshness(profile, config),
        concentration=_concentration(profile, config),
        win_anomaly=win_anom_score,
        timing=_timing(profile, config),
        size_odds=_size_odds(profile, config),
        single_purpose=_single_purpose(profile, config),
        sensitive_markets=_sensitive_markets(profile, config),
        cluster=_cluster(profile, config),
    )
    composite = (
        breakdown.freshness         * config.w_freshness
        + breakdown.concentration     * config.w_concentration
        + breakdown.win_anomaly       * config.w_win_anomaly
        + breakdown.timing            * config.w_timing
        + breakdown.size_odds         * config.w_size_odds
        + breakdown.single_purpose    * config.w_single_purpose
        + breakdown.sensitive_markets * config.w_sensitive_markets
        + breakdown.cluster           * config.w_cluster
    )
    composite = min(ONE, max(ZERO, composite)).quantize(Decimal("0.0001"))
    return InsiderScoreResult(
        composite=composite,
        win_rate_p_value=pv,
        breakdown=breakdown,
        classification=classification_from_score(composite, config),
    )
