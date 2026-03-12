"""Synthetic data generator for testing and development.

Generates random walk price series with known process noise (Q) and
measurement noise (R), providing ground truth for filter validation.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from loguru import logger

from src.data.models import PriceHistory, PricePoint
from src.utils.transforms import clip_probability

# Random seed for reproducibility in tests (set to None for random)
DEFAULT_SEED: int | None = 42


@dataclass
class SyntheticMarket:
    """Synthetic market data with known ground truth.

    Parameters
    ----------
    true_states : np.ndarray
        True underlying probabilities (unobserved in practice).
    observations : np.ndarray
        Noisy observed prices (what the market shows).
    timestamps : np.ndarray
        Array of datetime objects for each observation.
    Q_true : float
        True process noise variance used to generate the data.
    R_true : float
        True measurement noise variance used to generate the data.
    market_id : str
        Synthetic market identifier.
    question : str
        Descriptive label.
    """

    true_states: np.ndarray
    observations: np.ndarray
    timestamps: np.ndarray
    Q_true: float
    R_true: float
    market_id: str
    question: str


def generate_random_walk(
    n_steps: int = 500,
    Q: float = 1e-4,
    R: float = 1e-3,
    x0: float = 0.5,
    dt_hours: float = 1.0,
    seed: int | None = DEFAULT_SEED,
    market_id: str = "synthetic_random_walk",
    question: str = "Synthetic random walk market",
) -> SyntheticMarket:
    """Generate a random walk with Gaussian noise in probability space.

    The true state follows: x_t = clip(x_{t-1} + w_t), w_t ~ N(0, Q)
    Observations are: z_t = clip(x_t + v_t), v_t ~ N(0, R)

    Parameters
    ----------
    n_steps : int
        Number of time steps to generate.
    Q : float
        Process noise variance (true value).
    R : float
        Measurement noise variance (true value).
    x0 : float
        Initial true probability.
    dt_hours : float
        Hours between observations.
    seed : int or None
        Random seed for reproducibility. None for random.
    market_id : str
        Identifier for the synthetic market.
    question : str
        Label for the synthetic market.

    Returns
    -------
    SyntheticMarket
        Synthetic data with known ground truth.

    Examples
    --------
    >>> data = generate_random_walk(n_steps=100, Q=1e-4, R=1e-3, seed=42)
    >>> len(data.observations)
    100
    >>> 0.0 <= data.observations.min() and data.observations.max() <= 1.0
    True
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros(n_steps)
    observations = np.zeros(n_steps)
    true_states[0] = x0

    for t in range(1, n_steps):
        process_noise = rng.normal(0, np.sqrt(Q))
        true_states[t] = np.clip(true_states[t - 1] + process_noise, 0.0, 1.0)

    for t in range(n_steps):
        measurement_noise = rng.normal(0, np.sqrt(R))
        observations[t] = np.clip(true_states[t] + measurement_noise, 0.0, 1.0)

    start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = np.array([
        start_time + timedelta(hours=t * dt_hours) for t in range(n_steps)
    ])

    logger.debug(
        "Generated synthetic data: {} steps, Q={:.2e}, R={:.2e}",
        n_steps, Q, R,
    )

    return SyntheticMarket(
        true_states=true_states,
        observations=observations,
        timestamps=timestamps,
        Q_true=Q,
        R_true=R,
        market_id=market_id,
        question=question,
    )


def generate_step_change(
    n_steps: int = 500,
    Q: float = 1e-5,
    R: float = 1e-3,
    step_time: int = 250,
    step_from: float = 0.3,
    step_to: float = 0.7,
    seed: int | None = DEFAULT_SEED,
) -> SyntheticMarket:
    """Generate data with a sudden step change in the true state.

    Useful for testing adaptive filters and regime detection.

    Parameters
    ----------
    n_steps : int
        Total number of time steps.
    Q : float
        Process noise variance (small — the state is mostly stable).
    R : float
        Measurement noise variance.
    step_time : int
        Time step at which the jump occurs.
    step_from : float
        True probability before the jump.
    step_to : float
        True probability after the jump.
    seed : int or None
        Random seed.

    Returns
    -------
    SyntheticMarket
        Synthetic data with a step change at step_time.
    """
    rng = np.random.default_rng(seed)

    true_states = np.zeros(n_steps)
    observations = np.zeros(n_steps)

    for t in range(n_steps):
        base = step_from if t < step_time else step_to
        if t == 0:
            true_states[t] = base
        else:
            process_noise = rng.normal(0, np.sqrt(Q))
            true_states[t] = np.clip(base + process_noise, 0.0, 1.0)

        measurement_noise = rng.normal(0, np.sqrt(R))
        observations[t] = np.clip(true_states[t] + measurement_noise, 0.0, 1.0)

    start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = np.array([
        start_time + timedelta(hours=t) for t in range(n_steps)
    ])

    return SyntheticMarket(
        true_states=true_states,
        observations=observations,
        timestamps=timestamps,
        Q_true=Q,
        R_true=R,
        market_id="synthetic_step_change",
        question=f"Step change: {step_from:.1f} -> {step_to:.1f} at t={step_time}",
    )


def generate_sine_wave(
    n_steps: int = 500,
    R: float = 1e-3,
    amplitude: float = 0.2,
    center: float = 0.5,
    period_steps: int = 100,
    seed: int | None = DEFAULT_SEED,
) -> SyntheticMarket:
    """Generate a noisy sine wave in probability space.

    Useful for testing filter tracking of smooth changes.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    R : float
        Measurement noise variance.
    amplitude : float
        Sine wave amplitude (must keep center ± amplitude in [0, 1]).
    center : float
        Center probability value.
    period_steps : int
        Number of steps for one full sine cycle.
    seed : int or None
        Random seed.

    Returns
    -------
    SyntheticMarket
        Synthetic data following a noisy sine pattern.
    """
    rng = np.random.default_rng(seed)

    t_array = np.arange(n_steps)
    true_states = center + amplitude * np.sin(2.0 * np.pi * t_array / period_steps)
    true_states = np.clip(true_states, 0.0, 1.0)

    noise = rng.normal(0, np.sqrt(R), size=n_steps)
    observations = np.clip(true_states + noise, 0.0, 1.0)

    start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
    timestamps = np.array([
        start_time + timedelta(hours=t) for t in range(n_steps)
    ])

    # Q_true is approximate — sine wave is not a random walk
    approx_Q = (amplitude * 2 * np.pi / period_steps) ** 2

    return SyntheticMarket(
        true_states=true_states,
        observations=observations,
        timestamps=timestamps,
        Q_true=approx_Q,
        R_true=R,
        market_id="synthetic_sine",
        question=f"Sine wave: center={center}, amplitude={amplitude}",
    )


def synthetic_to_price_history(data: SyntheticMarket) -> PriceHistory:
    """Convert synthetic market data to PriceHistory format.

    Parameters
    ----------
    data : SyntheticMarket
        Synthetic data to convert.

    Returns
    -------
    PriceHistory
        Price history using the noisy observations.
    """
    points = [
        PricePoint(timestamp=ts, price=float(p))
        for ts, p in zip(data.timestamps, data.observations)
    ]
    return PriceHistory(
        market_id=data.market_id,
        question=data.question,
        points=points,
    )
