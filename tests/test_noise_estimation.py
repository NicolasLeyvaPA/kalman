"""Tests for dynamic observation noise estimation."""

import pytest

from src.data.models import MarketObservation
from src.filters.noise_estimation import (
    MAX_OBSERVATION_NOISE,
    MIN_OBSERVATION_NOISE,
    compute_depth_noise,
    compute_imbalance_noise,
    compute_observation_noise,
    compute_spread_noise,
    compute_stale_noise,
)
from datetime import datetime, timezone


def _make_observation(**kwargs) -> MarketObservation:
    """Helper to create a MarketObservation with defaults."""
    defaults = {
        "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc),
        "market_id": "test",
        "market_question": "Test?",
        "yes_price": 0.5,
        "spread": 0.02,
        "total_depth": 10000.0,
        "imbalance": 0.5,
        "num_trades_1h": 50,
    }
    defaults.update(kwargs)
    return MarketObservation(**defaults)


class TestNoiseEstimation:
    """Tests for observation noise R_t computation."""

    def test_spread_noise_increases_with_spread(self) -> None:
        """Wider spread should produce more noise."""
        r_narrow = compute_spread_noise(0.01)
        r_wide = compute_spread_noise(0.10)
        assert r_wide > r_narrow

    def test_spread_noise_zero_for_zero_spread(self) -> None:
        """Zero spread should give zero spread noise."""
        assert compute_spread_noise(0.0) == 0.0

    def test_depth_noise_increases_for_thin_book(self) -> None:
        """Thinner book should produce more noise."""
        r_thin = compute_depth_noise(100.0)
        r_deep = compute_depth_noise(100000.0)
        assert r_thin > r_deep

    def test_imbalance_noise_zero_at_balance(self) -> None:
        """Balanced book (imbalance=0.5) should give zero imbalance noise."""
        assert compute_imbalance_noise(0.5) == 0.0

    def test_imbalance_noise_symmetric(self) -> None:
        """Imbalance noise should be symmetric around 0.5."""
        r_low = compute_imbalance_noise(0.3)
        r_high = compute_imbalance_noise(0.7)
        assert abs(r_low - r_high) < 1e-10

    def test_stale_noise_decreases_with_activity(self) -> None:
        """More trades should produce less stale noise."""
        r_inactive = compute_stale_noise(0)
        r_active = compute_stale_noise(100)
        assert r_inactive > r_active

    def test_composite_noise_bounded(self) -> None:
        """Composite R_t should be within [MIN, MAX] bounds."""
        obs = _make_observation(spread=0.5, total_depth=1.0, imbalance=0.9, num_trades_1h=0)
        R = compute_observation_noise(obs)
        assert R <= MAX_OBSERVATION_NOISE
        assert R >= MIN_OBSERVATION_NOISE

    def test_liquid_market_low_noise(self) -> None:
        """Liquid market with tight spread and deep book should have low noise."""
        obs = _make_observation(
            spread=0.005, total_depth=500000.0, imbalance=0.5, num_trades_1h=200,
        )
        R = compute_observation_noise(obs)
        assert R < 1e-3

    def test_illiquid_market_high_noise(self) -> None:
        """Illiquid market should have high noise."""
        obs = _make_observation(
            spread=0.15, total_depth=50.0, imbalance=0.85, num_trades_1h=1,
        )
        R = compute_observation_noise(obs)
        assert R > 1e-3
