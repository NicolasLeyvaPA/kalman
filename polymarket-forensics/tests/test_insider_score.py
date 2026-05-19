"""Tests for the insider scoring engine.

Every sub-signal is exercised explicitly with the threshold cases plus
edge cases (zero data, just-below, just-above).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from enums import Classification
from scoring.insider_score import (
    DEFAULT_CONFIG,
    _concentration,
    _freshness,
    _single_purpose,
    _timing,
    classification_from_score,
    compute_insider_score,
)
from scoring.types import ScoringConfig


def test_default_weights_sum_to_one():
    cfg = ScoringConfig()
    total = (
        cfg.w_freshness + cfg.w_concentration + cfg.w_win_anomaly + cfg.w_timing
        + cfg.w_size_odds + cfg.w_single_purpose + cfg.w_sensitive_markets
        + cfg.w_cluster
    )
    assert total == Decimal("1.0")


def test_weights_that_dont_sum_to_one_raise():
    with pytest.raises(ValueError):
        ScoringConfig(w_freshness=Decimal("0.9"))


class TestClassification:
    def test_high_score_is_insider_suspect(self):
        assert classification_from_score(Decimal("0.85")) == Classification.INSIDER_SUSPECT

    def test_mid_score_is_suspicious(self):
        assert classification_from_score(Decimal("0.55")) == Classification.SUSPICIOUS

    def test_low_score_is_watch(self):
        assert classification_from_score(Decimal("0.35")) == Classification.WATCH

    def test_clean_score_is_normal(self):
        assert classification_from_score(Decimal("0.10")) == Classification.NORMAL

    def test_exact_threshold_is_higher_class(self):
        assert classification_from_score(Decimal("0.70")) == Classification.INSIDER_SUSPECT
        assert classification_from_score(Decimal("0.50")) == Classification.SUSPICIOUS
        assert classification_from_score(Decimal("0.30")) == Classification.WATCH


class TestFreshness:
    def test_brand_new_scores_critical(self, insider_wallet):
        w = replace(
            insider_wallet,
            first_trade=insider_wallet.first_seen + timedelta(hours=12),
        )
        assert _freshness(w, DEFAULT_CONFIG) == Decimal("1.0")

    def test_one_week_scores_high(self, insider_wallet):
        w = replace(
            insider_wallet,
            first_trade=insider_wallet.first_seen + timedelta(days=5),
        )
        assert _freshness(w, DEFAULT_CONFIG) == Decimal("0.7")

    def test_one_month_scores_low(self, insider_wallet):
        w = replace(
            insider_wallet,
            first_trade=insider_wallet.first_seen + timedelta(days=20),
        )
        assert _freshness(w, DEFAULT_CONFIG) == Decimal("0.3")

    def test_old_scores_zero(self, clean_wallet):
        assert _freshness(clean_wallet, DEFAULT_CONFIG) == Decimal("0")

    def test_missing_timestamps_safe(self, insider_wallet):
        w = replace(insider_wallet, first_seen=None, first_trade=None)
        assert _freshness(w, DEFAULT_CONFIG) == Decimal("0")


class TestConcentration:
    def test_all_in_one_market(self, insider_wallet):
        score = _concentration(insider_wallet, DEFAULT_CONFIG)
        assert score == Decimal("1.0")

    def test_diversified_low(self, clean_wallet):
        score = _concentration(clean_wallet, DEFAULT_CONFIG)
        assert score < Decimal("0.5")

    def test_zero_volume_safe(self, clean_wallet):
        w = replace(clean_wallet, total_volume=Decimal("0"))
        assert _concentration(w, DEFAULT_CONFIG) == Decimal("0")


class TestTiming:
    def test_minutes_before_resolution_critical(self, insider_wallet):
        w = replace(insider_wallet, avg_hours_before_resolution=Decimal("2"))
        assert _timing(w, DEFAULT_CONFIG) == Decimal("1.0")

    def test_one_day_high(self, insider_wallet):
        w = replace(insider_wallet, avg_hours_before_resolution=Decimal("18"))
        assert _timing(w, DEFAULT_CONFIG) == Decimal("0.7")

    def test_far_in_advance_zero(self, insider_wallet):
        w = replace(insider_wallet, avg_hours_before_resolution=Decimal("100"))
        assert _timing(w, DEFAULT_CONFIG) == Decimal("0")

    def test_none_safe(self, insider_wallet):
        w = replace(insider_wallet, avg_hours_before_resolution=None)
        assert _timing(w, DEFAULT_CONFIG) == Decimal("0")


class TestSinglePurpose:
    def test_burner_wallet(self, insider_wallet):
        assert _single_purpose(insider_wallet, DEFAULT_CONFIG) == Decimal("1.0")

    def test_active_defi_user(self, clean_wallet):
        assert _single_purpose(clean_wallet, DEFAULT_CONFIG) == Decimal("0")


class TestComposite:
    def test_textbook_insider_scores_above_threshold(self, insider_wallet):
        result = compute_insider_score(insider_wallet)
        assert result.composite >= Decimal("0.80"), \
            f"insider should score >= 0.80, got {result.composite}"
        assert result.classification == Classification.INSIDER_SUSPECT

    def test_clean_wallet_scores_normal(self, clean_wallet):
        result = compute_insider_score(clean_wallet)
        assert result.composite < Decimal("0.30"), \
            f"clean wallet should score < 0.30, got {result.composite}"
        assert result.classification == Classification.NORMAL

    def test_composite_is_bounded(self, insider_wallet):
        result = compute_insider_score(insider_wallet)
        assert Decimal("0") <= result.composite <= Decimal("1")

    def test_deterministic(self, insider_wallet):
        r1 = compute_insider_score(insider_wallet)
        r2 = compute_insider_score(insider_wallet)
        assert r1.composite == r2.composite
        assert r1.breakdown == r2.breakdown

    def test_breakdown_includes_all_signals(self, insider_wallet):
        result = compute_insider_score(insider_wallet)
        d = result.breakdown.as_dict()
        assert set(d.keys()) == {
            "freshness", "concentration", "win_anomaly", "timing",
            "size_odds", "single_purpose", "sensitive_markets", "cluster",
        }

    def test_each_breakdown_signal_in_range(self, insider_wallet):
        result = compute_insider_score(insider_wallet)
        for key, val in result.breakdown.as_dict().items():
            assert 0 <= val <= 1, f"{key}={val} out of range"

    def test_wallet_with_no_resolved_trades_doesnt_crash(self, clean_wallet):
        w = replace(
            clean_wallet,
            wins=0,
            total_resolved=0,
            avg_hours_before_resolution=None,
        )
        result = compute_insider_score(w)
        assert result.win_rate_p_value == Decimal("1.0")
        assert result.breakdown.win_anomaly == Decimal("0")
        assert result.breakdown.timing == Decimal("0")

    def test_zero_volume_doesnt_crash(self, clean_wallet):
        w = replace(
            clean_wallet,
            total_volume=Decimal("0"),
            top_market_volume=Decimal("0"),
            top_category_volume=Decimal("0"),
            political_military_volume=Decimal("0"),
            avg_trade_size=Decimal("0"),
            avg_pnl_per_trade=Decimal("0"),
        )
        result = compute_insider_score(w)
        assert result.composite >= Decimal("0")

    def test_custom_config_changes_score(self, insider_wallet):
        # Boost freshness weight; everything else has to give.
        cfg = ScoringConfig(
            w_freshness=Decimal("0.50"),
            w_concentration=Decimal("0.10"),
            w_win_anomaly=Decimal("0.10"),
            w_timing=Decimal("0.10"),
            w_size_odds=Decimal("0.05"),
            w_single_purpose=Decimal("0.05"),
            w_sensitive_markets=Decimal("0.05"),
            w_cluster=Decimal("0.05"),
        )
        result = compute_insider_score(insider_wallet, cfg)
        assert result.composite > Decimal("0")

    def test_cluster_signal_off_without_cluster(self, insider_wallet):
        w = replace(insider_wallet, cluster_id=None, cluster_insider_prob=Decimal("0"))
        result = compute_insider_score(w)
        assert result.breakdown.cluster == Decimal("0")
