"""Cross-market covariance estimation with Ledoit-Wolf shrinkage.

Estimates the process noise covariance matrix Q for the multivariate
Kalman filter from historical price returns. Uses Ledoit-Wolf shrinkage
to stabilize the sample covariance estimate, which is crucial when the
number of markets is comparable to or larger than the number of observations.

References
----------
Ledoit, O. & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix."
    Journal of Portfolio Management, 30(4), 110-119.
"""

import numpy as np
import pandas as pd
from loguru import logger

from src.utils.math_helpers import EPSILON


def compute_returns(prices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute first differences (returns) for each market's price series.

    Parameters
    ----------
    prices : dict[str, np.ndarray]
        Market ID -> price array. All arrays must have the same length.

    Returns
    -------
    dict[str, np.ndarray]
        Market ID -> returns array (length T-1).
    """
    return {
        market_id: np.diff(price_array)
        for market_id, price_array in prices.items()
    }


def sample_covariance(returns: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """Compute the sample covariance matrix of returns.

    Parameters
    ----------
    returns : dict[str, np.ndarray]
        Market ID -> returns array. All arrays must have the same length.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (covariance_matrix, market_ids) — the n×n covariance matrix and
        the ordered list of market IDs corresponding to rows/columns.
    """
    market_ids = sorted(returns.keys())
    n = len(market_ids)

    # Stack returns into a T × n matrix
    returns_matrix = np.column_stack([returns[mid] for mid in market_ids])
    cov = np.cov(returns_matrix, rowvar=False)

    # np.cov returns a scalar for n=1
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    return cov, market_ids


def ledoit_wolf_shrinkage(S: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Ledoit-Wolf shrinkage to a sample covariance matrix.

    Shrinks the sample covariance toward a scaled identity matrix.
    The optimal shrinkage intensity is estimated from the data.

    Parameters
    ----------
    S : np.ndarray
        Sample covariance matrix of shape (n, n).

    Returns
    -------
    tuple[np.ndarray, float]
        (shrunk_covariance, shrinkage_intensity) — the regularized
        covariance matrix and the estimated optimal shrinkage parameter alpha.

    Notes
    -----
    The shrinkage target is F = mu * I, where mu = trace(S) / n.
    The shrunk covariance is: S_shrunk = alpha * F + (1 - alpha) * S
    where alpha in [0, 1] is the optimal shrinkage intensity.

    Reference: Ledoit & Wolf (2004), simplified estimator.
    """
    n = S.shape[0]
    if n == 1:
        return S.copy(), 0.0

    # Target: scaled identity
    mu = np.trace(S) / n
    F = mu * np.eye(n)

    # Estimate optimal shrinkage intensity
    # Using the Oracle Approximating Shrinkage (OAS) estimator
    delta = S - F
    delta_sq_sum = np.sum(delta ** 2)
    trace_S_sq = np.trace(S @ S)

    # Simplified Ledoit-Wolf formula
    numerator = delta_sq_sum
    denominator = (n + 1 - 2.0 / n) * (trace_S_sq + mu ** 2 * n) - 2.0 * mu * np.trace(S)

    if abs(denominator) < EPSILON:
        alpha = 1.0
    else:
        alpha = float(np.clip(numerator / denominator, 0.0, 1.0))

    S_shrunk = alpha * F + (1.0 - alpha) * S

    logger.debug(
        "Ledoit-Wolf shrinkage: alpha={:.4f}, n={}", alpha, n,
    )
    return S_shrunk, alpha


def estimate_cross_covariance(
    market_prices: dict[str, np.ndarray],
    scaling: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    """Estimate the process noise covariance Q from price returns.

    Parameters
    ----------
    market_prices : dict[str, np.ndarray]
        Market ID -> price array. All arrays must have the same length.
    scaling : float
        Scaling factor applied to the covariance. Use to match the
        time scale of your filter updates.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        (Q_matrix, market_ids) — the estimated n×n process noise
        covariance matrix and the ordered list of market IDs.
    """
    returns = compute_returns(market_prices)
    S, market_ids = sample_covariance(returns)
    Q_shrunk, alpha = ledoit_wolf_shrinkage(S)
    Q = Q_shrunk * scaling

    logger.info(
        "Estimated Q: {}x{}, shrinkage alpha={:.4f}, max diag={:.2e}",
        Q.shape[0], Q.shape[1], alpha, np.max(np.diag(Q)),
    )
    return Q, market_ids


def correlation_matrix(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        Correlation matrix of shape (n, n), with ones on the diagonal.
    """
    stds = np.sqrt(np.diag(cov))
    outer = np.outer(stds, stds)
    outer = np.maximum(outer, EPSILON)
    return cov / outer
