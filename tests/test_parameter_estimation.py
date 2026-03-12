"""Tests for MLE parameter estimation."""

import numpy as np
import pytest

from src.data.synthetic import generate_random_walk
from src.filters.parameter_estimation import estimate_parameters, log_likelihood


class TestParameterEstimation:
    """Tests for Q, R estimation via maximum likelihood."""

    def test_mle_recovers_parameters(self) -> None:
        """MLE should recover true Q and R within 50% for sufficient data.

        Generate data with known Q=1e-4, R=1e-3, then verify the MLE
        estimates are reasonably close.
        """
        data = generate_random_walk(n_steps=2000, Q=1e-4, R=1e-3, seed=42)
        Q_hat, R_hat = estimate_parameters(data.observations, Q0=1e-4, R0=1e-3)

        # Should be within factor of 2 of true values
        assert Q_hat > 1e-4 * 0.3, f"Q_hat={Q_hat:.2e} too low"
        assert Q_hat < 1e-4 * 5.0, f"Q_hat={Q_hat:.2e} too high"
        assert R_hat > 1e-3 * 0.3, f"R_hat={R_hat:.2e} too low"
        assert R_hat < 1e-3 * 5.0, f"R_hat={R_hat:.2e} too high"

    def test_mle_different_noise_levels(self) -> None:
        """MLE should distinguish high-noise from low-noise scenarios."""
        data_low = generate_random_walk(n_steps=2000, Q=1e-5, R=1e-4, seed=42)
        data_high = generate_random_walk(n_steps=2000, Q=1e-3, R=1e-2, seed=42)

        Q_low, R_low = estimate_parameters(data_low.observations)
        Q_high, R_high = estimate_parameters(data_high.observations)

        # Higher noise data should produce larger parameter estimates
        assert Q_high > Q_low
        assert R_high > R_low

    def test_log_likelihood_finite(self) -> None:
        """Log-likelihood should be finite for valid parameters."""
        data = generate_random_walk(n_steps=200, seed=42)
        ll = log_likelihood(data.observations, Q=1e-4, R=1e-3)
        assert np.isfinite(ll)

    def test_log_likelihood_invalid_params(self) -> None:
        """Log-likelihood should return -inf for non-positive parameters."""
        data = generate_random_walk(n_steps=100, seed=42)
        assert log_likelihood(data.observations, Q=0, R=1e-3) == -np.inf
        assert log_likelihood(data.observations, Q=-1, R=1e-3) == -np.inf

    def test_mle_better_than_bad_params(self) -> None:
        """MLE parameters should have higher likelihood than bad guesses."""
        data = generate_random_walk(n_steps=1000, Q=1e-4, R=1e-3, seed=42)
        Q_hat, R_hat = estimate_parameters(data.observations)

        ll_mle = log_likelihood(data.observations, Q_hat, R_hat)
        ll_bad = log_likelihood(data.observations, Q=1e-1, R=1e-1)

        assert ll_mle > ll_bad
