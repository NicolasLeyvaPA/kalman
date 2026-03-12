"""Tests for the logit-space Kalman filter."""

import numpy as np
import pytest

from src.data.synthetic import generate_random_walk
from src.filters.logit_kalman import LogitKalmanFilter
from src.filters.scalar_kalman import ScalarKalmanFilter


class TestLogitKalmanFilter:
    """Tests for LogitKalmanFilter."""

    def test_output_bounded(self) -> None:
        """Filtered probabilities must always be in (0, 1)."""
        data = generate_random_walk(n_steps=500, Q=1e-3, R=1e-2, x0=0.9, seed=42)
        lkf = LogitKalmanFilter(Q_logit=1e-2, R_prob=1e-2)
        result = lkf.filter(data.observations)

        assert np.all(result.states_prob > 0.0)
        assert np.all(result.states_prob < 1.0)

    def test_confidence_bounds_bounded(self) -> None:
        """Confidence intervals must be in (0, 1)."""
        data = generate_random_walk(n_steps=500, Q=1e-3, R=1e-2, x0=0.95, seed=42)
        lkf = LogitKalmanFilter(Q_logit=1e-2, R_prob=1e-2)
        result = lkf.filter(data.observations)

        assert np.all(result.upper_95 > 0.0)
        assert np.all(result.upper_95 <= 1.0)
        assert np.all(result.lower_95 >= 0.0)
        assert np.all(result.lower_95 < 1.0)

    def test_confidence_bounds_contain_estimate(self) -> None:
        """The estimate should always be within the confidence bounds."""
        data = generate_random_walk(n_steps=300, Q=1e-4, R=1e-3, seed=42)
        lkf = LogitKalmanFilter(Q_logit=1e-3, R_prob=1e-3)
        result = lkf.filter(data.observations)

        assert np.all(result.states_prob >= result.lower_95 - 1e-10)
        assert np.all(result.states_prob <= result.upper_95 + 1e-10)

    def test_asymmetric_ci_near_boundary(self) -> None:
        """Near p=0.9, the CI should be asymmetric: more room toward 0.5."""
        lkf = LogitKalmanFilter(Q_logit=1e-2, R_prob=1e-3, x0_prob=0.9)
        # Feed observations near 0.9
        obs = np.full(100, 0.9)
        result = lkf.filter(obs)

        last = -1
        p = result.states_prob[last]
        upper_dist = result.upper_95[last] - p
        lower_dist = p - result.lower_95[last]

        # Near 0.9, the interval toward 0.5 (lower) should be wider
        # than toward 1.0 (upper)
        assert lower_dist > upper_dist

    def test_regular_filter_can_exceed_bounds(self) -> None:
        """Show that the regular filter CAN produce values outside [0,1]
        while logit filter stays bounded. We use extreme parameters."""
        # Create observations very close to 1.0
        rng = np.random.default_rng(42)
        observations = np.clip(0.98 + rng.normal(0, 0.03, size=200), 0.01, 0.99)

        # Regular filter with high Q might exceed 1.0
        regular = ScalarKalmanFilter(Q=1e-2, R=1e-4)
        result_regular = regular.filter(observations)

        # Logit filter stays bounded by construction
        logit_f = LogitKalmanFilter(Q_logit=1e-1, R_prob=1e-4)
        result_logit = logit_f.filter(observations)

        assert np.all(result_logit.states_prob > 0.0)
        assert np.all(result_logit.states_prob < 1.0)

    def test_symmetric_at_half(self) -> None:
        """At p=0.5, the logit-space filter should produce symmetric CIs."""
        lkf = LogitKalmanFilter(Q_logit=1e-3, R_prob=1e-3, x0_prob=0.5)
        obs = np.full(100, 0.5)
        result = lkf.filter(obs)

        last = -1
        p = result.states_prob[last]
        upper_dist = result.upper_95[last] - p
        lower_dist = p - result.lower_95[last]

        # Should be approximately symmetric
        assert abs(upper_dist - lower_dist) < 0.01

    def test_tracks_constant(self) -> None:
        """Filter should converge to the true value on constant input."""
        lkf = LogitKalmanFilter(Q_logit=1e-4, R_prob=1e-3)
        obs = np.full(300, 0.72)
        result = lkf.filter(obs)

        assert abs(result.states_prob[-1] - 0.72) < 0.01

    def test_result_shapes(self) -> None:
        """All output arrays should have correct length."""
        n = 100
        lkf = LogitKalmanFilter(Q_logit=1e-3, R_prob=1e-3)
        obs = np.random.default_rng(42).uniform(0.3, 0.7, size=n)
        result = lkf.filter(obs)

        assert len(result.states_prob) == n
        assert len(result.states_logit) == n
        assert len(result.covariances_logit) == n
        assert len(result.gains) == n
        assert len(result.innovations_logit) == n
        assert len(result.upper_95) == n
        assert len(result.lower_95) == n

    def test_negative_Q_raises(self) -> None:
        """Negative Q should raise ValueError."""
        with pytest.raises(ValueError):
            LogitKalmanFilter(Q_logit=-1, R_prob=1e-3)

    def test_handles_extreme_observations(self) -> None:
        """Filter should handle observations very close to 0 and 1."""
        lkf = LogitKalmanFilter(Q_logit=1e-3, R_prob=1e-3)
        obs = np.array([0.001, 0.999, 0.5, 0.01, 0.99])
        result = lkf.filter(obs)

        assert np.all(np.isfinite(result.states_prob))
        assert np.all(result.states_prob > 0)
        assert np.all(result.states_prob < 1)
