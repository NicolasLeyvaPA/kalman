"""Maximum likelihood estimation of Kalman filter parameters.

Estimates optimal process noise (Q) and measurement noise (R) from
observation data by maximizing the log-likelihood of the innovation sequence.

The innovation sequence of a correctly specified Kalman filter is Gaussian
white noise with zero mean and covariance S_t. The log-likelihood is:

    log L = -0.5 * sum_t [ log(2*pi*S_t) + y_t^2 / S_t ]

where y_t is the innovation and S_t is the innovation covariance at time t.

References
----------
Schweppe, F.C. (1965). "Evaluation of Likelihood Functions for Gaussian Signals."
IEEE Transactions on Information Theory, 11(1), 61-70.
"""

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from src.filters.scalar_kalman import ScalarKalmanFilter
from src.utils.math_helpers import EPSILON

# Optimization bounds for Q and R (in log space)
LOG_Q_BOUNDS: tuple[float, float] = (-10.0, 0.0)  # Q from 1e-10 to 1
LOG_R_BOUNDS: tuple[float, float] = (-10.0, 0.0)  # R from 1e-10 to 1

# Number of initial warm-up steps to skip when computing likelihood
# (the filter is adapting during these steps, so their contribution is unreliable)
WARMUP_STEPS: int = 10


def log_likelihood(observations: np.ndarray, Q: float, R: float) -> float:
    """Compute the log-likelihood of observations given filter parameters.

    Runs the Kalman filter and computes the log-likelihood from the
    innovation sequence. Higher values indicate better model fit.

    Parameters
    ----------
    observations : np.ndarray
        Array of observed market prices.
    Q : float
        Process noise variance.
    R : float
        Measurement noise variance.

    Returns
    -------
    float
        Log-likelihood value. More positive = better fit.

    Notes
    -----
    log L = -0.5 * sum_t [ log(2*pi*S_t) + y_t^2 / S_t ]

    We skip the first WARMUP_STEPS observations to avoid the transient
    period where the filter is still converging.
    """
    if Q <= 0 or R <= 0:
        return -np.inf

    kf = ScalarKalmanFilter(Q=Q, R=R)
    result = kf.filter(observations)

    # Skip warm-up period
    start = min(WARMUP_STEPS, len(observations) - 1)
    innovations = result.innovations[start:]
    S = result.innovation_covariances[start:]

    # Avoid log(0) and division by zero
    S = np.maximum(S, EPSILON)

    ll = -0.5 * np.sum(np.log(2.0 * np.pi * S) + innovations ** 2 / S)
    return float(ll)


def _negative_log_likelihood(log_params: np.ndarray,
                              observations: np.ndarray) -> float:
    """Negative log-likelihood as a function of log-transformed parameters.

    Used as the objective function for scipy.optimize.minimize.

    Parameters
    ----------
    log_params : np.ndarray
        [log(Q), log(R)] — log-transformed parameters for unconstrained optimization.
    observations : np.ndarray
        Observed prices.

    Returns
    -------
    float
        Negative log-likelihood (to be minimized).
    """
    Q = np.exp(log_params[0])
    R = np.exp(log_params[1])
    ll = log_likelihood(observations, Q, R)
    return -ll


def estimate_parameters(
    observations: np.ndarray,
    Q0: float = 1e-4,
    R0: float = 1e-3,
    method: str = "L-BFGS-B",
) -> tuple[float, float]:
    """Estimate optimal Q and R via maximum likelihood.

    Uses scipy.optimize.minimize on the negative log-likelihood.
    Optimization is performed in log-space to enforce positivity.

    Parameters
    ----------
    observations : np.ndarray
        Array of observed market prices.
    Q0 : float
        Initial guess for process noise variance.
    R0 : float
        Initial guess for measurement noise variance.
    method : str
        Optimization method. Default "L-BFGS-B" supports bounds.

    Returns
    -------
    tuple[float, float]
        (Q_hat, R_hat) — estimated optimal parameters.

    Examples
    --------
    >>> from src.data.synthetic import generate_random_walk
    >>> data = generate_random_walk(n_steps=1000, Q=1e-4, R=1e-3, seed=42)
    >>> Q_hat, R_hat = estimate_parameters(data.observations)
    """
    log_params0 = np.array([np.log(Q0), np.log(R0)])
    bounds = [LOG_Q_BOUNDS, LOG_R_BOUNDS]

    result = minimize(
        _negative_log_likelihood,
        log_params0,
        args=(observations,),
        method=method,
        bounds=bounds,
    )

    Q_hat = float(np.exp(result.x[0]))
    R_hat = float(np.exp(result.x[1]))

    logger.info(
        "MLE estimation: Q={:.2e}, R={:.2e} (converged={}, nll={:.2f})",
        Q_hat, R_hat, result.success, result.fun,
    )
    return Q_hat, R_hat


def likelihood_surface(
    observations: np.ndarray,
    Q_range: np.ndarray,
    R_range: np.ndarray,
) -> np.ndarray:
    """Compute the log-likelihood over a grid of Q and R values.

    Useful for visualizing the likelihood landscape as a heatmap.

    Parameters
    ----------
    observations : np.ndarray
        Array of observed market prices.
    Q_range : np.ndarray
        Array of Q values to evaluate.
    R_range : np.ndarray
        Array of R values to evaluate.

    Returns
    -------
    np.ndarray
        2D array of log-likelihood values, shape (len(Q_range), len(R_range)).
    """
    grid = np.zeros((len(Q_range), len(R_range)))
    for i, Q in enumerate(Q_range):
        for j, R in enumerate(R_range):
            grid[i, j] = log_likelihood(observations, Q, R)

    logger.debug(
        "Computed likelihood surface: {}x{} grid, max ll={:.2f}",
        len(Q_range), len(R_range), grid.max(),
    )
    return grid
