"""Scalar Kalman filter for probability estimation.

Implements the basic Kalman filter for a scalar state (single probability)
with a random walk state transition model. This is the foundation that
all other filter variants build upon.

State-space model:
    State transition:  x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)
    Observation:       z_t = x_t + v_t,       v_t ~ N(0, R)

References
----------
Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
Journal of Basic Engineering, 82(1), 35-45.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.utils.math_helpers import DEFAULT_INITIAL_COVARIANCE, EPSILON


@dataclass
class KalmanState:
    """Complete state of the Kalman filter after one predict-update cycle.

    Parameters
    ----------
    x : float
        State estimate (filtered probability).
    P : float
        State covariance (estimation uncertainty).
    K : float
        Kalman gain used in this update (0 to 1).
    innovation : float
        Innovation (prediction error): z - x_prior.
    z : float
        Raw observation used in this update.
    x_prior : float
        Predicted state before incorporating the observation.
    P_prior : float
        Predicted covariance before the update.
    S : float
        Innovation covariance: P_prior + R.
    """

    x: float
    P: float
    K: float
    innovation: float
    z: float
    x_prior: float = 0.0
    P_prior: float = 0.0
    S: float = 0.0


@dataclass
class KalmanResult:
    """Results from running the filter over an entire observation sequence.

    Parameters
    ----------
    states : np.ndarray
        Filtered probability estimates at each time step.
    covariances : np.ndarray
        Estimation uncertainty at each time step.
    gains : np.ndarray
        Kalman gain at each time step.
    innovations : np.ndarray
        Innovation (prediction error) at each time step.
    observations : np.ndarray
        Raw observations (for comparison).
    timestamps : np.ndarray or None
        Timestamps for each observation, if provided.
    innovation_covariances : np.ndarray
        Innovation covariance S_t at each time step.
    predicted_states : np.ndarray
        Prior (predicted) state before each update.
    """

    states: np.ndarray
    covariances: np.ndarray
    gains: np.ndarray
    innovations: np.ndarray
    observations: np.ndarray
    timestamps: np.ndarray | None = None
    innovation_covariances: np.ndarray = None  # type: ignore[assignment]
    predicted_states: np.ndarray = None  # type: ignore[assignment]


class ScalarKalmanFilter:
    """Kalman filter for scalar state estimation.

    Estimates the true underlying probability of a prediction market event
    from noisy observed prices. Uses a random walk model for the state
    transition (the true probability is expected to change slowly).

    Parameters
    ----------
    Q : float
        Process noise variance — how much the true probability can change
        between observations. Larger Q = model expects faster changes.
        Typical range: 1e-6 to 1e-3.
    R : float
        Measurement noise variance — how noisy the observed price is.
        Larger R = less trust in observations.
        Typical range: 1e-4 to 1e-2.
    x0 : float or None
        Initial state estimate. If None, uses the first observation.
    P0 : float
        Initial state uncertainty. Default 0.25 (maximum entropy for [0,1]).

    Examples
    --------
    >>> kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
    >>> state = kf.step(0.55)
    >>> print(f"Estimate: {state.x:.4f}, Gain: {state.K:.4f}")
    """

    def __init__(self, Q: float, R: float,
                 x0: float | None = None,
                 P0: float = DEFAULT_INITIAL_COVARIANCE) -> None:
        if Q < 0:
            raise ValueError(f"Process noise Q must be non-negative, got {Q}")
        if R < 0:
            raise ValueError(f"Measurement noise R must be non-negative, got {R}")

        self.Q = Q
        self.R = R
        self.x: float = x0 if x0 is not None else 0.0
        self.P: float = P0
        self._initialized: bool = x0 is not None
        self._step_count: int = 0

        logger.debug("ScalarKalmanFilter initialized: Q={:.2e}, R={:.2e}", Q, R)

    def predict(self) -> tuple[float, float]:
        """Prediction step: project state and covariance forward.

        For the random walk model, the predicted state equals the current
        state (no drift), and uncertainty grows by the process noise Q.

        Returns
        -------
        tuple[float, float]
            (predicted_state, predicted_covariance).

        Notes
        -----
        x_prior = x_hat  (random walk: E[x_t | x_{t-1}] = x_{t-1})
        P_prior = P + Q  (uncertainty grows by process noise)
        """
        x_prior = self.x
        P_prior = self.P + self.Q
        return x_prior, P_prior

    def update(self, z: float, x_prior: float | None = None,
               P_prior: float | None = None) -> tuple[float, float, float, float, float]:
        """Update step: incorporate a new observation.

        Parameters
        ----------
        z : float
            New observation (market price).
        x_prior : float or None
            Predicted state. If None, uses the current stored state.
        P_prior : float or None
            Predicted covariance. If None, uses P + Q.

        Returns
        -------
        tuple[float, float, float, float, float]
            (updated_state, updated_covariance, kalman_gain, innovation,
             innovation_covariance).

        Notes
        -----
        Innovation:       y = z - x_prior
        Innovation cov:   S = P_prior + R
        Kalman gain:      K = P_prior / S
        Updated state:    x = x_prior + K * y
        Updated cov:      P = (1 - K) * P_prior
        """
        if x_prior is None:
            x_prior = self.x
        if P_prior is None:
            P_prior = self.P + self.Q

        # Innovation (prediction error)
        innovation = z - x_prior

        # Innovation covariance
        S = P_prior + self.R

        # Kalman gain
        K = P_prior / (S + EPSILON)

        # State update
        x_post = x_prior + K * innovation

        # Covariance update (Joseph form is more stable but overkill for scalar)
        P_post = (1.0 - K) * P_prior

        # Ensure covariance stays positive
        P_post = max(P_post, EPSILON)

        return x_post, P_post, K, innovation, S

    def step(self, z: float) -> KalmanState:
        """Execute one full predict-update cycle.

        If this is the first observation and no initial state was provided,
        the filter initializes to this observation.

        Parameters
        ----------
        z : float
            New observation (market price).

        Returns
        -------
        KalmanState
            Complete filter state after the update.
        """
        self._step_count += 1

        # Initialize on first observation if no x0 was given
        if not self._initialized:
            self.x = z
            self._initialized = True
            logger.debug("Filter initialized at z={:.4f}", z)

        # Predict
        x_prior, P_prior = self.predict()

        # Update
        x_post, P_post, K, innovation, S = self.update(z, x_prior, P_prior)

        # Store updated state
        self.x = x_post
        self.P = P_post

        return KalmanState(
            x=x_post,
            P=P_post,
            K=K,
            innovation=innovation,
            z=z,
            x_prior=x_prior,
            P_prior=P_prior,
            S=S,
        )

    def filter(self, observations: np.ndarray,
               timestamps: np.ndarray | None = None) -> KalmanResult:
        """Run the filter over an entire observation sequence.

        Parameters
        ----------
        observations : np.ndarray
            Array of observed prices (length T).
        timestamps : np.ndarray or None
            Optional array of timestamps (length T).

        Returns
        -------
        KalmanResult
            Complete filter output including states, covariances, gains,
            and innovations for each time step.
        """
        n = len(observations)
        states = np.zeros(n)
        covariances = np.zeros(n)
        gains = np.zeros(n)
        innovations = np.zeros(n)
        innovation_covs = np.zeros(n)
        predicted_states = np.zeros(n)

        for t in range(n):
            state = self.step(observations[t])
            states[t] = state.x
            covariances[t] = state.P
            gains[t] = state.K
            innovations[t] = state.innovation
            innovation_covs[t] = state.S
            predicted_states[t] = state.x_prior

        logger.info(
            "Filtered {} observations. Final state: x={:.4f}, P={:.2e}, K={:.4f}",
            n, self.x, self.P, gains[-1] if n > 0 else 0.0,
        )

        return KalmanResult(
            states=states,
            covariances=covariances,
            gains=gains,
            innovations=innovations,
            observations=observations.copy(),
            timestamps=timestamps,
            innovation_covariances=innovation_covs,
            predicted_states=predicted_states,
        )

    def reset(self, x0: float | None = None,
              P0: float = DEFAULT_INITIAL_COVARIANCE) -> None:
        """Reset the filter to initial conditions.

        Parameters
        ----------
        x0 : float or None
            New initial state. None means re-initialize on first observation.
        P0 : float
            New initial covariance.
        """
        self.x = x0 if x0 is not None else 0.0
        self.P = P0
        self._initialized = x0 is not None
        self._step_count = 0
