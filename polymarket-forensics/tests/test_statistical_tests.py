"""Tests for the statistical-test primitives."""

from __future__ import annotations

from decimal import Decimal

import pytest

from exceptions import InsufficientDataError
from scoring.statistical_tests import volume_anomaly_z_score, win_rate_p_value


class TestWinRatePValue:
    def test_no_trades_returns_one(self):
        assert win_rate_p_value(0, 0, Decimal("0.5")) == Decimal("1.0")

    def test_no_wins_returns_one(self):
        assert win_rate_p_value(0, 10, Decimal("0.5")) == Decimal("1.0")

    def test_wins_exceeding_total_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            win_rate_p_value(11, 10, Decimal("0.5"))

    def test_perfect_streak_at_fair_odds_is_extreme(self):
        """14 wins out of 14 at 50/50 odds = p < 1e-4."""
        pv = win_rate_p_value(14, 14, Decimal("0.5"))
        assert pv < Decimal("0.0001")

    def test_perfect_streak_at_long_odds_is_astronomical(self):
        """14/14 at 18% implied probability ~ 1e-11."""
        pv = win_rate_p_value(14, 14, Decimal("0.18"))
        assert pv < Decimal("1e-10")

    def test_normal_winrate_is_not_significant(self):
        """6 wins out of 10 at 50% odds is unremarkable."""
        pv = win_rate_p_value(6, 10, Decimal("0.5"))
        assert pv > Decimal("0.05")

    def test_clamps_extreme_implied_probability(self):
        # Implied prob outside [0.001, 0.999] is clamped.
        pv_high = win_rate_p_value(5, 10, Decimal("0.9999"))
        pv_low = win_rate_p_value(5, 10, Decimal("0.0001"))
        assert pv_high > Decimal("0")
        assert pv_low > Decimal("0")

    def test_returned_value_is_decimal(self):
        result = win_rate_p_value(5, 10, Decimal("0.5"))
        assert isinstance(result, Decimal)


class TestVolumeAnomalyZScore:
    def test_basic_z_score(self):
        z = volume_anomaly_z_score(
            current_volume=Decimal("100"),
            baseline_mean=Decimal("50"),
            baseline_std=Decimal("10"),
        )
        assert z == Decimal("5")

    def test_zero_std_raises(self):
        with pytest.raises(InsufficientDataError):
            volume_anomaly_z_score(
                current_volume=Decimal("100"),
                baseline_mean=Decimal("50"),
                baseline_std=Decimal("0"),
            )

    def test_negative_std_raises(self):
        with pytest.raises(InsufficientDataError):
            volume_anomaly_z_score(
                current_volume=Decimal("100"),
                baseline_mean=Decimal("50"),
                baseline_std=Decimal("-1"),
            )
