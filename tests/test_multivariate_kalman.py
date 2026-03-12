"""Tests for the multivariate Kalman filter."""

import numpy as np
import pytest

from src.filters.multivariate_kalman import MultivariateKalmanFilter
from src.filters.scalar_kalman import ScalarKalmanFilter


class TestMultivariateKalmanFilter:
    """Tests for MultivariateKalmanFilter."""

    def test_independent_markets(self) -> None:
        """With diagonal Q, multivariate filter should match n independent scalar filters.

        This is the fundamental sanity check: when markets are uncorrelated,
        the multivariate filter degenerates to independent scalar filters.
        """
        n = 3
        Q_scalar = 1e-4
        R_scalar = 1e-3

        Q = np.diag([Q_scalar] * n)
        R = np.diag([R_scalar] * n)

        rng = np.random.default_rng(42)
        T = 200
        observations = rng.uniform(0.3, 0.7, size=(T, n))

        # Multivariate filter
        mkf = MultivariateKalmanFilter(n=n, Q=Q, R=R)
        result_mv = mkf.filter(observations)

        # Independent scalar filters
        for j in range(n):
            skf = ScalarKalmanFilter(Q=Q_scalar, R=R_scalar)
            result_s = skf.filter(observations[:, j])
            np.testing.assert_allclose(
                result_mv.states[:, j], result_s.states, atol=1e-10,
                err_msg=f"Market {j} mismatch",
            )

    def test_cross_correlation_propagation(self) -> None:
        """Observing only market 1 should shift market 2 via correlation.

        Two positively correlated markets. We observe only market 1.
        Market 2's estimate should shift in the same direction.
        """
        Q = np.array([[1e-4, 8e-5], [8e-5, 1e-4]])  # Strong positive correlation
        R = np.diag([1e-3, 1e-3])

        mkf = MultivariateKalmanFilter(
            n=2, Q=Q, R=R,
            x0=np.array([0.5, 0.5]),
        )

        # Observe only market 0 going up
        z = np.array([0.7, 0.0])  # Market 1 value doesn't matter
        mask = np.array([True, False])
        state = mkf.update(z, observed_mask=mask)

        # Market 0 should increase toward 0.7
        assert state.x[0] > 0.5

        # Market 1 should ALSO increase due to positive correlation
        assert state.x[1] > 0.5

    def test_negative_correlation_propagation(self) -> None:
        """Negative correlation: observing market 0 up should push market 1 down."""
        Q = np.array([[1e-4, -8e-5], [-8e-5, 1e-4]])  # Negative correlation
        R = np.diag([1e-3, 1e-3])

        mkf = MultivariateKalmanFilter(
            n=2, Q=Q, R=R,
            x0=np.array([0.5, 0.5]),
        )

        z = np.array([0.7, 0.0])
        mask = np.array([True, False])
        state = mkf.update(z, observed_mask=mask)

        assert state.x[0] > 0.5
        assert state.x[1] < 0.5  # Pushed down by negative correlation

    def test_missing_observations(self) -> None:
        """Filter should handle NaN observations (missing markets)."""
        Q = np.diag([1e-4, 1e-4])
        R = np.diag([1e-3, 1e-3])

        T = 50
        observations = np.full((T, 2), 0.5)
        observations[10:20, 1] = np.nan  # Market 1 unobserved for 10 steps

        mkf = MultivariateKalmanFilter(n=2, Q=Q, R=R)
        result = mkf.filter(observations)

        # Should complete without error
        assert result.states.shape == (T, 2)
        assert np.all(np.isfinite(result.states))

    def test_covariance_positive_definite(self) -> None:
        """Covariance should remain positive definite throughout."""
        Q = np.array([[1e-4, 5e-5], [5e-5, 1e-4]])
        R = np.diag([1e-3, 1e-3])

        rng = np.random.default_rng(42)
        observations = rng.uniform(0.3, 0.7, size=(100, 2))

        mkf = MultivariateKalmanFilter(n=2, Q=Q, R=R)
        result = mkf.filter(observations)

        for t in range(100):
            eigenvalues = np.linalg.eigvalsh(result.covariances[t])
            assert np.all(eigenvalues > 0), f"Non-PD covariance at step {t}"

    def test_single_market_matches_scalar(self) -> None:
        """n=1 multivariate filter should match the scalar filter."""
        Q_val = 1e-4
        R_val = 1e-3

        rng = np.random.default_rng(42)
        observations_1d = rng.uniform(0.3, 0.7, size=100)
        observations_2d = observations_1d.reshape(-1, 1)

        mkf = MultivariateKalmanFilter(
            n=1, Q=np.array([[Q_val]]), R=np.array([[R_val]]),
        )
        result_mv = mkf.filter(observations_2d)

        skf = ScalarKalmanFilter(Q=Q_val, R=R_val)
        result_s = skf.filter(observations_1d)

        np.testing.assert_allclose(
            result_mv.states[:, 0], result_s.states, atol=1e-10,
        )

    def test_no_observations_just_predicts(self) -> None:
        """When no markets are observed, filter should just predict."""
        Q = np.diag([1e-4, 1e-4])
        R = np.diag([1e-3, 1e-3])

        mkf = MultivariateKalmanFilter(
            n=2, Q=Q, R=R, x0=np.array([0.5, 0.5]),
        )

        z = np.array([0.0, 0.0])
        mask = np.array([False, False])
        state = mkf.update(z, observed_mask=mask)

        # State should not change (just random walk predict = no change in mean)
        np.testing.assert_allclose(state.x, [0.5, 0.5])

    def test_invalid_dimensions_raise(self) -> None:
        """Mismatched Q dimensions should raise ValueError."""
        with pytest.raises(ValueError):
            MultivariateKalmanFilter(
                n=2, Q=np.eye(3), R=np.eye(2),
            )

    def test_three_market_filter(self) -> None:
        """Filter should work with 3 markets."""
        Q = np.eye(3) * 1e-4
        Q[0, 1] = Q[1, 0] = 5e-5
        R = np.eye(3) * 1e-3

        rng = np.random.default_rng(42)
        obs = rng.uniform(0.3, 0.7, size=(50, 3))

        mkf = MultivariateKalmanFilter(n=3, Q=Q, R=R)
        result = mkf.filter(obs)

        assert result.states.shape == (50, 3)
        assert np.all(np.isfinite(result.states))
