"""Unit tests for the scalar Kalman filter.

Tests cover convergence, innovation properties, covariance positivity,
steady-state gain, and tracking of synthetic signals.
"""

import numpy as np
import pytest
from scipy import stats

from src.data.synthetic import generate_random_walk, generate_sine_wave, generate_step_change
from src.filters.scalar_kalman import KalmanResult, KalmanState, ScalarKalmanFilter
from src.utils.math_helpers import steady_state_gain


class TestScalarKalmanFilter:
    """Tests for ScalarKalmanFilter."""

    def test_constant_signal(self) -> None:
        """Filter on constant observations should converge to that constant."""
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        observations = np.full(200, 0.65)
        result = kf.filter(observations)

        # After convergence, estimate should be very close to 0.65
        assert abs(result.states[-1] - 0.65) < 1e-4

    def test_step_change(self) -> None:
        """Filter should track a step change in the true state."""
        data = generate_step_change(
            n_steps=500, step_time=250,
            step_from=0.3, step_to=0.7, Q=1e-5, R=1e-3, seed=42,
        )
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        result = kf.filter(data.observations)

        # Before step: estimate should be near 0.3
        assert abs(result.states[200] - 0.3) < 0.05

        # After step and adaptation: estimate should be near 0.7
        assert abs(result.states[-1] - 0.7) < 0.05

    def test_noisy_sine(self) -> None:
        """Filter on a noisy sine wave should produce smoother output."""
        data = generate_sine_wave(n_steps=500, R=1e-2, seed=42)
        kf = ScalarKalmanFilter(Q=1e-3, R=1e-2)
        result = kf.filter(data.observations)

        # Filtered output should be smoother (lower variance of differences)
        raw_var = np.var(np.diff(data.observations))
        filtered_var = np.var(np.diff(result.states))
        assert filtered_var < raw_var

    def test_gain_convergence(self) -> None:
        """Kalman gain should converge to a steady-state value."""
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3, x0=0.5)
        observations = np.random.default_rng(42).normal(0.5, 0.03, size=500)
        result = kf.filter(observations)

        # Gains in last 50 steps should be nearly constant
        late_gains = result.gains[-50:]
        gain_variation = np.std(late_gains)
        assert gain_variation < 1e-6

    def test_innovation_whiteness(self) -> None:
        """Innovation sequence should be approximately white noise.

        For a correctly specified model, innovations should be uncorrelated.
        We test this using the Ljung-Box test on the first 10 lags.
        """
        data = generate_random_walk(n_steps=2000, Q=1e-4, R=1e-3, seed=42)
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        result = kf.filter(data.observations)

        # Skip the first few innovations during filter warm-up
        innovations = result.innovations[50:]

        # Autocorrelation at lag 1 should be small
        autocorr = np.corrcoef(innovations[:-1], innovations[1:])[0, 1]
        assert abs(autocorr) < 0.1

    def test_innovation_within_bounds(self) -> None:
        """~95% of normalized innovations should be within +/-2.

        The normalized innovation y_t / sqrt(S_t) should be approximately N(0,1)
        under correct model specification.
        """
        data = generate_random_walk(n_steps=2000, Q=1e-4, R=1e-3, seed=42)
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        result = kf.filter(data.observations)

        # Skip warm-up
        innovations = result.innovations[50:]
        S = result.innovation_covariances[50:]
        normalized = innovations / np.sqrt(S)

        fraction_within = np.mean(np.abs(normalized) < 2.0)
        # Should be close to 95.4% (within 2 sigma for normal)
        assert fraction_within > 0.90
        assert fraction_within < 0.99

    def test_covariance_positive(self) -> None:
        """State covariance should always remain positive."""
        data = generate_random_walk(n_steps=1000, Q=1e-4, R=1e-3, seed=42)
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        result = kf.filter(data.observations)

        assert np.all(result.covariances > 0)

    def test_known_solution(self) -> None:
        """Verify steady-state gain matches the algebraic Riccati solution.

        For constant Q, R, after many steps, the Kalman gain converges to:
        K_ss = (-Q + sqrt(Q^2 + 4*Q*R)) / (2*R) * 1/(1 - that_ratio)
        More precisely: P_ss = (-Q + sqrt(Q^2 + 4*Q*R)) / 2, K_ss = P_ss / (P_ss + R)
        """
        Q, R = 1e-4, 1e-3
        kf = ScalarKalmanFilter(Q=Q, R=R, x0=0.5)
        observations = np.full(1000, 0.5)
        result = kf.filter(observations)

        K_theoretical = steady_state_gain(Q, R)
        K_numerical = result.gains[-1]

        assert abs(K_theoretical - K_numerical) < 1e-6

    def test_initialization_from_first_observation(self) -> None:
        """Without x0, filter should initialize from the first observation."""
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        state = kf.step(0.73)
        # After first step, state should be close to the observation
        assert abs(state.x - 0.73) < 0.1

    def test_high_Q_tracks_fast(self) -> None:
        """High Q should make the filter more responsive to changes."""
        data = generate_step_change(
            n_steps=200, step_time=100,
            step_from=0.3, step_to=0.7, Q=1e-6, R=1e-3, seed=42,
        )

        # High Q filter
        kf_high = ScalarKalmanFilter(Q=1e-2, R=1e-3)
        result_high = kf_high.filter(data.observations)

        # Low Q filter
        kf_low = ScalarKalmanFilter(Q=1e-6, R=1e-3)
        result_low = kf_low.filter(data.observations)

        # At step 120 (20 after change), high Q should be closer to 0.7
        error_high = abs(result_high.states[120] - 0.7)
        error_low = abs(result_low.states[120] - 0.7)
        assert error_high < error_low

    def test_high_R_smooths_more(self) -> None:
        """High R should produce smoother (less responsive) estimates."""
        data = generate_random_walk(n_steps=500, Q=1e-4, R=1e-2, seed=42)

        kf_low_R = ScalarKalmanFilter(Q=1e-4, R=1e-4)
        result_low = kf_low_R.filter(data.observations)

        kf_high_R = ScalarKalmanFilter(Q=1e-4, R=1e-1)
        result_high = kf_high_R.filter(data.observations)

        # High R should produce smoother output
        var_low = np.var(np.diff(result_low.states))
        var_high = np.var(np.diff(result_high.states))
        assert var_high < var_low

    def test_filter_result_shapes(self) -> None:
        """All output arrays should have the same length as input."""
        n = 100
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
        observations = np.random.default_rng(42).uniform(0.3, 0.7, size=n)
        result = kf.filter(observations)

        assert len(result.states) == n
        assert len(result.covariances) == n
        assert len(result.gains) == n
        assert len(result.innovations) == n
        assert len(result.observations) == n
        assert len(result.innovation_covariances) == n

    def test_negative_Q_raises(self) -> None:
        """Negative Q should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ScalarKalmanFilter(Q=-1e-4, R=1e-3)

    def test_negative_R_raises(self) -> None:
        """Negative R should raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            ScalarKalmanFilter(Q=1e-4, R=-1e-3)

    def test_reset(self) -> None:
        """Reset should return the filter to initial conditions."""
        kf = ScalarKalmanFilter(Q=1e-4, R=1e-3, x0=0.5)
        _ = kf.step(0.6)
        _ = kf.step(0.7)

        kf.reset(x0=0.5)
        assert kf.x == 0.5
        assert kf._step_count == 0
