"""Numerical stability helpers and mathematical utilities.

Provides safe division, matrix operations, and constants used throughout
the filter implementations.
"""

import numpy as np

# Small constant added before divisions to prevent division by zero
EPSILON: float = 1e-10

# Default initial state covariance — represents high uncertainty
DEFAULT_INITIAL_COVARIANCE: float = 0.25


def safe_divide(numerator: np.ndarray | float,
                denominator: np.ndarray | float) -> np.ndarray | float:
    """Divide with epsilon protection against division by zero.

    Parameters
    ----------
    numerator : np.ndarray or float
        Numerator value(s).
    denominator : np.ndarray or float
        Denominator value(s). EPSILON is added to avoid zero division.

    Returns
    -------
    np.ndarray or float
        Result of numerator / (denominator + EPSILON).
    """
    return numerator / (denominator + EPSILON)


def ensure_positive_definite(matrix: np.ndarray,
                             min_eigenvalue: float = EPSILON) -> np.ndarray:
    """Ensure a symmetric matrix is positive definite.

    If any eigenvalue is below min_eigenvalue, it is clamped up. This prevents
    numerical issues in the Kalman filter covariance updates.

    Parameters
    ----------
    matrix : np.ndarray
        Symmetric matrix of shape (n, n).
    min_eigenvalue : float
        Minimum allowed eigenvalue. Default is EPSILON.

    Returns
    -------
    np.ndarray
        Positive definite matrix of shape (n, n).
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Force a matrix to be exactly symmetric.

    Numerical errors can make covariance matrices slightly asymmetric.
    This corrects that by averaging with the transpose.

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix of shape (n, n).

    Returns
    -------
    np.ndarray
        Symmetric matrix: (matrix + matrix.T) / 2.
    """
    return 0.5 * (matrix + matrix.T)


def steady_state_gain(Q: float, R: float) -> float:
    """Compute the steady-state Kalman gain for a scalar filter.

    For a scalar random walk model with constant Q and R, the Kalman gain
    converges to a fixed value as the number of observations grows.

    Parameters
    ----------
    Q : float
        Process noise variance.
    R : float
        Measurement noise variance.

    Returns
    -------
    float
        Steady-state Kalman gain K_ss.

    Notes
    -----
    P_ss = (Q + sqrt(Q^2 + 4*Q*R)) / 2, then K_ss = P_ss / (P_ss + R)

    Derived from the algebraic Riccati equation for the scalar case:
    P_ss = P_ss + Q - P_ss^2 / (P_ss + R)
    K_ss = P_ss / (P_ss + R)
    """
    discriminant = Q ** 2 + 4.0 * Q * R
    P_ss = (Q + np.sqrt(discriminant)) / 2.0
    return safe_divide(P_ss, P_ss + R)
