"""Tests for the adaptive Kalman filter."""

import numpy as np
import pytest

from src.data.synthetic import generate_random_walk, generate_step_change
from src.filters.adaptive_kalman import AdaptiveKalmanFilter
from src.filters.scalar_kalman import ScalarKalmanFilter


class TestAdaptiveKalmanFilter:
    """Tests for AdaptiveKalmanFilter."""

    def test_no_inflation_on_stable_signal(self) -> None:
        """On a stable signal, Q should remain at baseline."""
        data = generate_random_walk(n_steps=200, Q=1e-5, R=1e-3, seed=42)
        akf = AdaptiveKalmanFilter(Q_base=1e-4, R=1e-3)
        akf.filter(data.observations)

        Q_hist = akf.get_Q_history()
        # Most Q values should be at or near baseline
        at_baseline = np.sum(np.isclose(Q_hist, 1e-4, rtol=0.1))
        assert at_baseline > len(Q_hist) * 0.8

    def test_inflation_on_step_change(self) -> None:
        """Q should inflate after a step change."""
        data = generate_step_change(
            n_steps=300, step_time=150,
            step_from=0.3, step_to=0.7, Q=1e-6, R=1e-3, seed=42,
        )
        akf = AdaptiveKalmanFilter(
            Q_base=1e-5, R=1e-3, threshold=2.0, inflation=20.0,
        )
        akf.filter(data.observations)

        Q_hist = akf.get_Q_history()
        # Should see inflated Q around the step change
        max_Q = Q_hist.max()
        assert max_Q > 1e-5 * 5  # At least 5x inflation

    def test_decay_after_inflation(self) -> None:
        """Q inflation should decay back toward baseline."""
        akf = AdaptiveKalmanFilter(
            Q_base=1e-4, R=1e-3, threshold=2.0, inflation=10.0, decay=0.5, x0=0.5,
        )

        # Feed a big surprise
        akf.step(0.5)
        state = akf.step(0.9)  # Large innovation
        Q_after_surprise = akf.Q_effective

        # Feed normal observations and watch Q decay
        Q_values = []
        for _ in range(20):
            akf.step(0.5)
            Q_values.append(akf.Q_effective)

        # Q should have decayed
        assert Q_values[-1] < Q_after_surprise

    def test_adaptive_tracks_step_faster(self) -> None:
        """Adaptive filter should track a step change faster than basic filter."""
        data = generate_step_change(
            n_steps=300, step_time=150,
            step_from=0.3, step_to=0.7, Q=1e-6, R=1e-3, seed=42,
        )

        # Basic filter
        basic = ScalarKalmanFilter(Q=1e-5, R=1e-3)
        result_basic = basic.filter(data.observations)

        # Adaptive filter
        adaptive = AdaptiveKalmanFilter(
            Q_base=1e-5, R=1e-3, threshold=2.0, inflation=20.0,
        )
        result_adaptive = adaptive.filter(data.observations)

        # At step 170 (20 steps after change), adaptive should be closer to 0.7
        error_basic = abs(result_basic.states[170] - 0.7)
        error_adaptive = abs(result_adaptive.states[170] - 0.7)
        assert error_adaptive < error_basic

    def test_dynamic_R_accepted(self) -> None:
        """Filter should accept and use time-varying R_t values."""
        data = generate_random_walk(n_steps=100, seed=42)
        R_t_array = np.full(100, 1e-3)
        R_t_array[50:] = 1e-2  # Noisier second half

        akf = AdaptiveKalmanFilter(Q_base=1e-4, R=1e-3)
        result = akf.filter(data.observations, R_t_array=R_t_array)

        R_hist = akf.get_R_history()
        assert np.isclose(R_hist[25], 1e-3)
        assert np.isclose(R_hist[75], 1e-2)

    def test_filter_result_shapes(self) -> None:
        """Output arrays should match input length."""
        n = 100
        akf = AdaptiveKalmanFilter(Q_base=1e-4, R=1e-3)
        obs = np.random.default_rng(42).uniform(0.3, 0.7, size=n)
        result = akf.filter(obs)

        assert len(result.states) == n
        assert len(result.covariances) == n
        assert len(result.gains) == n
        assert len(result.innovations) == n

    def test_invalid_decay_raises(self) -> None:
        """Decay outside (0,1) should raise ValueError."""
        with pytest.raises(ValueError, match="Decay"):
            AdaptiveKalmanFilter(Q_base=1e-4, R=1e-3, decay=1.5)

    def test_reset(self) -> None:
        """Reset should clear inflation and history."""
        akf = AdaptiveKalmanFilter(Q_base=1e-4, R=1e-3, x0=0.5)
        akf.step(0.9)  # Trigger inflation
        akf.reset(x0=0.5)

        assert akf._current_inflation == 1.0
        assert len(akf._Q_history) == 0
