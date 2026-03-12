"""Probability transforms: logit, sigmoid, and related utilities.

These transforms map between probability space [0, 1] and logit space (-inf, +inf),
enabling the Kalman filter to operate in an unconstrained domain.
"""

import numpy as np

# Clipping bound to avoid log(0) or division by zero in logit transform
PROBABILITY_CLIP_EPSILON: float = 1e-10


def logit(p: np.ndarray | float) -> np.ndarray | float:
    """Transform probability to logit (log-odds) space.

    Parameters
    ----------
    p : np.ndarray or float
        Probability value(s) in [0, 1]. Values are clipped to
        [PROBABILITY_CLIP_EPSILON, 1 - PROBABILITY_CLIP_EPSILON] for numerical stability.

    Returns
    -------
    np.ndarray or float
        Logit-transformed value(s) in (-inf, +inf).

    Notes
    -----
    logit(p) = log(p / (1 - p))

    Examples
    --------
    >>> logit(0.5)
    0.0
    >>> logit(0.9)  # doctest: +ELLIPSIS
    2.197...
    """
    p_clipped = np.clip(p, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    return np.log(p_clipped / (1.0 - p_clipped))


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Transform logit-space value back to probability.

    Parameters
    ----------
    x : np.ndarray or float
        Logit-space value(s) in (-inf, +inf).

    Returns
    -------
    np.ndarray or float
        Probability value(s) in (0, 1).

    Notes
    -----
    sigmoid(x) = 1 / (1 + exp(-x))

    Uses a numerically stable implementation that avoids overflow for large |x|.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(2.197)  # doctest: +ELLIPSIS
    0.899...
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def logit_noise_transform(R_prob: float, p: float) -> float:
    """Transform observation noise variance from probability space to logit space.

    Uses the delta method: if z ~ N(p, R_prob) in probability space,
    then logit(z) ~ N(logit(p), R_logit) approximately, where
    R_logit = R_prob / (p * (1 - p))^2.

    Parameters
    ----------
    R_prob : float
        Observation noise variance in probability space.
    p : float
        Current probability estimate (observation value).

    Returns
    -------
    float
        Observation noise variance in logit space.

    Notes
    -----
    The delta method approximation: Var[g(X)] ≈ (g'(μ))^2 * Var[X]
    where g = logit, g'(p) = 1 / (p * (1 - p)).

    Near p = 0 or p = 1, the logit-space noise becomes very large, correctly
    reflecting that small probability-space changes near boundaries correspond
    to large logit-space changes.
    """
    p_clipped = np.clip(p, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON)
    jacobian_sq = 1.0 / (p_clipped * (1.0 - p_clipped)) ** 2
    return R_prob * jacobian_sq


def clip_probability(p: np.ndarray | float) -> np.ndarray | float:
    """Clip value(s) to valid probability range [0, 1].

    Parameters
    ----------
    p : np.ndarray or float
        Value(s) to clip.

    Returns
    -------
    np.ndarray or float
        Clipped value(s) in [0, 1].
    """
    return np.clip(p, 0.0, 1.0)
