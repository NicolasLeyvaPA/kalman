"""Evaluation metrics for filter performance.

Provides Brier score, log loss, calibration curve computation, and
innovation diagnostics for assessing filter accuracy.
"""

import numpy as np
from loguru import logger

from src.utils.math_helpers import EPSILON


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the Brier score: mean squared error of probability predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities in [0, 1].
    outcomes : np.ndarray
        Binary outcomes (0 or 1).

    Returns
    -------
    float
        Brier score. Lower is better. Range [0, 1].
        A score of 0.25 corresponds to always predicting 0.5 (no skill).
    """
    return float(np.mean((predictions - outcomes) ** 2))


def log_loss(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute the log loss (cross-entropy) of probability predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities in [0, 1]. Clipped to avoid log(0).
    outcomes : np.ndarray
        Binary outcomes (0 or 1).

    Returns
    -------
    float
        Log loss. Lower is better.
    """
    p = np.clip(predictions, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(outcomes * np.log(p) + (1.0 - outcomes) * np.log(1.0 - p)))


def calibration_curve(
    predictions: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a calibration curve (reliability diagram data).

    Groups predictions into bins and computes the actual outcome frequency
    in each bin. A perfectly calibrated predictor has bin_actual ≈ bin_predicted.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities.
    outcomes : np.ndarray
        Binary outcomes.
    n_bins : int
        Number of probability bins.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (bin_centers, bin_actual_freq, bin_counts) — the center of each bin,
        the actual outcome frequency, and the number of predictions in each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_actual = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            mask = mask | (predictions == bin_edges[i + 1])
        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            bin_actual[i] = outcomes[mask].mean()

    return bin_centers, bin_actual, bin_counts


def innovation_diagnostics(
    innovations: np.ndarray,
    innovation_covariances: np.ndarray,
) -> dict[str, float]:
    """Compute diagnostic statistics for the innovation sequence.

    Parameters
    ----------
    innovations : np.ndarray
        Innovation sequence y_t.
    innovation_covariances : np.ndarray
        Innovation covariance S_t at each step.

    Returns
    -------
    dict[str, float]
        Dictionary with:
        - 'mean': mean of innovations (should be ~0)
        - 'std': std of innovations
        - 'fraction_within_2sigma': fraction within ±2σ (should be ~0.95)
        - 'autocorrelation_lag1': lag-1 autocorrelation (should be ~0)
        - 'normalized_mean': mean of normalized innovations
        - 'normalized_std': std of normalized innovations (should be ~1)
    """
    S = np.maximum(innovation_covariances, EPSILON)
    normalized = innovations / np.sqrt(S)

    within_2sigma = np.mean(np.abs(normalized) < 2.0)

    # Lag-1 autocorrelation
    if len(innovations) > 2:
        autocorr = float(np.corrcoef(innovations[:-1], innovations[1:])[0, 1])
    else:
        autocorr = 0.0

    return {
        "mean": float(np.mean(innovations)),
        "std": float(np.std(innovations)),
        "fraction_within_2sigma": float(within_2sigma),
        "autocorrelation_lag1": autocorr,
        "normalized_mean": float(np.mean(normalized)),
        "normalized_std": float(np.std(normalized)),
    }
