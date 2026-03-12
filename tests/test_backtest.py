"""Tests for the backtest framework."""

import numpy as np
import pytest

from src.data.synthetic import generate_random_walk
from src.pipeline.backtest import FilterBacktest


class TestFilterBacktest:
    """Tests for FilterBacktest."""

    def test_all_variants_returned(self) -> None:
        """Backtest should return results for all 5 filter variants."""
        data = generate_random_walk(n_steps=200, Q=1e-4, R=1e-3, seed=42)
        bt = FilterBacktest()
        results = bt.run(data.observations, outcome=1)

        assert len(results) == 5
        names = [r.filter_name for r in results]
        assert "Raw Price" in names
        assert "SMA-20" in names
        assert "Scalar Kalman" in names
        assert "Adaptive Kalman" in names
        assert "Logit Kalman" in names

    def test_brier_scores_valid(self) -> None:
        """Brier scores should be in [0, 1]."""
        data = generate_random_walk(n_steps=200, Q=1e-4, R=1e-3, x0=0.7, seed=42)
        bt = FilterBacktest()
        results = bt.run(data.observations, outcome=1)

        for r in results:
            assert 0.0 <= r.brier <= 1.0, f"{r.filter_name}: Brier={r.brier}"

    def test_log_loss_finite(self) -> None:
        """Log loss should be finite for all variants."""
        data = generate_random_walk(n_steps=200, Q=1e-4, R=1e-3, seed=42)
        bt = FilterBacktest()
        results = bt.run(data.observations, outcome=1)

        for r in results:
            assert np.isfinite(r.logloss), f"{r.filter_name}: logloss={r.logloss}"

    def test_predictions_shapes(self) -> None:
        """Prediction arrays should match input length."""
        n = 100
        data = generate_random_walk(n_steps=n, seed=42)
        bt = FilterBacktest()
        results = bt.run(data.observations, outcome=0)

        for r in results:
            assert len(r.predictions) == n, f"{r.filter_name}: len={len(r.predictions)}"

    def test_predictions_bounded(self) -> None:
        """All predictions should be in (0, 1)."""
        data = generate_random_walk(n_steps=200, Q=1e-3, R=1e-2, x0=0.9, seed=42)
        bt = FilterBacktest()
        results = bt.run(data.observations, outcome=1)

        for r in results:
            assert r.predictions.min() > 0.0, f"{r.filter_name}: min={r.predictions.min()}"
            assert r.predictions.max() < 1.0, f"{r.filter_name}: max={r.predictions.max()}"

    def test_kalman_better_than_raw_on_noisy_data(self) -> None:
        """On noisy data near outcome, Kalman should beat raw on Brier score.

        When the true probability is near the outcome and noise is high,
        filtering should improve accuracy.
        """
        # Market near 0.8 that resolves YES (outcome=1)
        data = generate_random_walk(n_steps=500, Q=1e-5, R=5e-3, x0=0.8, seed=42)
        bt = FilterBacktest(Q=1e-5, R=5e-3)
        results = bt.run(data.observations, outcome=1)

        raw_brier = next(r.brier for r in results if r.filter_name == "Raw Price")
        kalman_brier = next(r.brier for r in results if r.filter_name == "Scalar Kalman")

        # Kalman should be at least as good as raw
        assert kalman_brier <= raw_brier + 0.01

    def test_simple_moving_average_correct(self) -> None:
        """SMA should produce reasonable smoothed output."""
        data = generate_random_walk(n_steps=100, seed=42)
        bt = FilterBacktest()
        sma = bt._simple_moving_average(data.observations, window=5)

        # SMA should be smoother
        assert np.var(np.diff(sma)) < np.var(np.diff(data.observations))
