"""Dynamic observation noise estimation from market microstructure.

Estimates time-varying measurement noise R_t from order book data:
spread width, book depth, order imbalance, and trade frequency.
Markets with wide spreads, thin books, or low activity produce
noisier price signals.

References
----------
Hasbrouck, J. (2007). "Empirical Market Microstructure." Oxford University Press.
"""

import numpy as np
from loguru import logger

from src.data.models import MarketObservation
from src.utils.math_helpers import EPSILON

# Component weights for the composite noise model.
# These control the relative importance of each noise source.
# Calibrate via cross-validation on held-out data.
WEIGHT_SPREAD: float = 0.4
WEIGHT_DEPTH: float = 0.25
WEIGHT_IMBALANCE: float = 0.2
WEIGHT_STALE: float = 0.15

# Scaling constant for depth-based noise: larger = more noise for thin books
DEPTH_SCALING: float = 1e-3

# Scaling constant for stale-quote noise: larger = more noise for inactive markets
STALE_SCALING: float = 5e-4

# Minimum noise floor to prevent filter from becoming overconfident
MIN_OBSERVATION_NOISE: float = 1e-6

# Maximum observation noise to prevent filter from ignoring all observations
MAX_OBSERVATION_NOISE: float = 0.1


def compute_observation_noise(obs: MarketObservation) -> float:
    """Estimate observation noise R_t from market microstructure data.

    Combines four noise components:
    1. Spread noise: wider spread = noisier midpoint
    2. Depth noise: thinner book = easier to push price
    3. Imbalance noise: asymmetric book = price being pushed one direction
    4. Stale noise: fewer trades = price may be outdated

    Parameters
    ----------
    obs : MarketObservation
        Current market observation with microstructure data.

    Returns
    -------
    float
        Estimated observation noise variance R_t.

    Notes
    -----
    R_t = w1*R_spread + w2*R_depth + w3*R_imbalance + w4*R_stale

    where:
        R_spread = (spread / 2)^2
        R_depth = DEPTH_SCALING / log(1 + total_depth)
        R_imbalance = (imbalance - 0.5)^2
        R_stale = STALE_SCALING / (1 + num_trades_1h)
    """
    R_spread = compute_spread_noise(obs.spread)
    R_depth = compute_depth_noise(obs.total_depth)
    R_imbalance = compute_imbalance_noise(obs.imbalance)
    R_stale = compute_stale_noise(obs.num_trades_1h)

    R_t = (
        WEIGHT_SPREAD * R_spread
        + WEIGHT_DEPTH * R_depth
        + WEIGHT_IMBALANCE * R_imbalance
        + WEIGHT_STALE * R_stale
    )

    R_t = np.clip(R_t, MIN_OBSERVATION_NOISE, MAX_OBSERVATION_NOISE)

    logger.debug(
        "R_t={:.2e} (spread={:.2e}, depth={:.2e}, imbal={:.2e}, stale={:.2e})",
        R_t, R_spread, R_depth, R_imbalance, R_stale,
    )
    return float(R_t)


def compute_spread_noise(spread: float) -> float:
    """Noise from bid-ask spread.

    The midpoint of a wide-spread market is uncertain by approximately
    half the spread. We use the variance of a uniform distribution
    over the spread as the noise estimate.

    Parameters
    ----------
    spread : float
        Bid-ask spread (best_ask - best_bid).

    Returns
    -------
    float
        Spread-based noise variance.
    """
    half_spread = max(spread, 0.0) / 2.0
    return half_spread ** 2


def compute_depth_noise(total_depth: float) -> float:
    """Noise from order book depth.

    Thin books can be moved by small orders, making the price unreliable.
    Noise decreases logarithmically with depth.

    Parameters
    ----------
    total_depth : float
        Total dollar depth (bid + ask sides) in the top levels.

    Returns
    -------
    float
        Depth-based noise variance.
    """
    return DEPTH_SCALING / (np.log(1.0 + max(total_depth, 0.0)) + EPSILON)


def compute_imbalance_noise(imbalance: float) -> float:
    """Noise from order book imbalance.

    When the book is heavily one-sided (imbalance far from 0.5), the
    midpoint is being pushed by directional pressure and is less reliable.

    Parameters
    ----------
    imbalance : float
        Book imbalance: bid_depth / total_depth. 0.5 = balanced.

    Returns
    -------
    float
        Imbalance-based noise variance.
    """
    deviation = abs(imbalance - 0.5)
    return deviation ** 2


def compute_stale_noise(num_trades_1h: int) -> float:
    """Noise from low trading activity.

    Markets with few recent trades may have stale quotes that don't
    reflect current information. Noise decreases with activity.

    Parameters
    ----------
    num_trades_1h : int
        Number of trades in the last hour.

    Returns
    -------
    float
        Stale-quote noise variance.
    """
    return STALE_SCALING / (1.0 + max(num_trades_1h, 0))
