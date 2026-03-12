"""Logit-space Kalman filter for bounded probability estimation.

Operates in logit space (log-odds) where the domain is unbounded (-inf, +inf),
avoiding the problem of the regular Kalman filter producing estimates
outside [0, 1]. Observations are transformed to logit space before filtering,
and estimates are transformed back to probability space via the sigmoid function.

The key insight is that in logit space:
- The state can take any real value without violating probability bounds
- Confidence intervals transform back to asymmetric bands in probability space
  (compressed near 0 and 1, wider near 0.5)
- The delta method handles the noise transformation:
  R_logit ≈ R_prob / (p * (1-p))^2
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.filters.scalar_kalman import KalmanResult, KalmanState, ScalarKalmanFilter
from src.utils.math_helpers import DEFAULT_INITIAL_COVARIANCE, EPSILON
from src.utils.transforms import (
    PROBABILITY_CLIP_EPSILON,
    clip_probability,
    logit,
    logit_noise_transform,
    sigmoid,
)


@dataclass
class LogitKalmanState:
    """Filter state with both logit-space and probability-space values.

    Parameters
    ----------
    x_prob : float
        State estimate in probability space [0, 1].
    x_logit : float
        State estimate in logit space.
    P_logit : float
        State covariance in logit space.
    K : float
        Kalman gain (in logit space).
    innovation_logit : float
        Innovation in logit space.
    z_prob : float
        Raw observation in probability space.
    z_logit : float
        Observation transformed to logit space.
    upper_95 : float
        Upper 95% confidence bound in probability space.
    lower_95 : float
        Lower 95% confidence bound in probability space.
    """

    x_prob: float
    x_logit: float
    P_logit: float
    K: float
    innovation_logit: float
    z_prob: float
    z_logit: float
    upper_95: float
    lower_95: float


@dataclass
class LogitKalmanResult:
    """Results from the logit-space filter.

    Parameters
    ----------
    states_prob : np.ndarray
        Filtered probabilities in [0, 1].
    states_logit : np.ndarray
        Filtered states in logit space.
    covariances_logit : np.ndarray
        Covariances in logit space.
    gains : np.ndarray
        Kalman gains (logit space).
    innovations_logit : np.ndarray
        Innovations in logit space.
    observations_prob : np.ndarray
        Raw observations in probability space.
    upper_95 : np.ndarray
        Upper 95% confidence bounds in probability space.
    lower_95 : np.ndarray
        Lower 95% confidence bounds in probability space.
    timestamps : np.ndarray or None
        Timestamps.
    innovation_covariances : np.ndarray
        Innovation covariance S_t in logit space.
    """

    states_prob: np.ndarray
    states_logit: np.ndarray
    covariances_logit: np.ndarray
    gains: np.ndarray
    innovations_logit: np.ndarray
    observations_prob: np.ndarray
    upper_95: np.ndarray
    lower_95: np.ndarray
    timestamps: np.ndarray | None = None
    innovation_covariances: np.ndarray = None  # type: ignore[assignment]


# Number of standard deviations for confidence bands
CONFIDENCE_SIGMA: float = 2.0


class LogitKalmanFilter:
    """Kalman filter operating in logit-transformed space.

    Transforms observations from probability space [0,1] to logit space (-inf, +inf),
    runs the standard Kalman filter there, and transforms estimates back.
    This ensures filtered probabilities are always in [0, 1].

    Parameters
    ----------
    Q_logit : float
        Process noise variance in logit space.
    R_prob : float
        Measurement noise variance in probability space. Transformed to logit
        space at each step using the delta method.
    x0_prob : float or None
        Initial probability estimate. Transformed to logit space internally.
    P0_logit : float
        Initial covariance in logit space.

    Notes
    -----
    The observation noise in logit space varies with the current price level:
        R_logit = R_prob / (p * (1-p))^2

    Near p=0.5, R_logit ≈ 16*R_prob (moderate)
    Near p=0.95, R_logit ≈ 400*R_prob (much noisier)
    This correctly captures that small price changes near boundaries
    correspond to large logit-space changes.

    Examples
    --------
    >>> lkf = LogitKalmanFilter(Q_logit=1e-3, R_prob=1e-3)
    >>> state = lkf.step(0.95)
    >>> print(f"Prob: {state.x_prob:.4f}, CI: [{state.lower_95:.4f}, {state.upper_95:.4f}]")
    """

    def __init__(
        self,
        Q_logit: float,
        R_prob: float,
        x0_prob: float | None = None,
        P0_logit: float = 1.0,
    ) -> None:
        if Q_logit < 0:
            raise ValueError(f"Q_logit must be non-negative, got {Q_logit}")
        if R_prob < 0:
            raise ValueError(f"R_prob must be non-negative, got {R_prob}")

        self.Q_logit = Q_logit
        self.R_prob = R_prob

        # State in logit space
        if x0_prob is not None:
            self.x_logit = float(logit(x0_prob))
        else:
            self.x_logit = 0.0  # logit(0.5)
        self.P_logit = P0_logit
        self._initialized = x0_prob is not None
        self._step_count = 0

        logger.debug(
            "LogitKalmanFilter: Q_logit={:.2e}, R_prob={:.2e}", Q_logit, R_prob,
        )

    def step(self, z_prob: float) -> LogitKalmanState:
        """Execute one predict-update cycle in logit space.

        Parameters
        ----------
        z_prob : float
            New observation in probability space [0, 1].

        Returns
        -------
        LogitKalmanState
            Complete state with both logit and probability space values.
        """
        self._step_count += 1

        # Clip and transform observation
        z_clipped = float(np.clip(z_prob, PROBABILITY_CLIP_EPSILON, 1.0 - PROBABILITY_CLIP_EPSILON))
        z_logit = float(logit(z_clipped))

        # Initialize on first observation
        if not self._initialized:
            self.x_logit = z_logit
            self._initialized = True

        # Transform observation noise to logit space via delta method
        R_logit = logit_noise_transform(self.R_prob, z_clipped)

        # Predict in logit space
        x_prior = self.x_logit
        P_prior = self.P_logit + self.Q_logit

        # Update in logit space
        innovation = z_logit - x_prior
        S = P_prior + R_logit
        K = P_prior / (S + EPSILON)
        x_post = x_prior + K * innovation
        P_post = (1.0 - K) * P_prior
        P_post = max(P_post, EPSILON)

        # Store logit-space state
        self.x_logit = x_post
        self.P_logit = P_post

        # Transform back to probability space
        x_prob = float(sigmoid(x_post))

        # Confidence interval in probability space (asymmetric)
        std_logit = np.sqrt(P_post) * CONFIDENCE_SIGMA
        upper_95 = float(sigmoid(x_post + std_logit))
        lower_95 = float(sigmoid(x_post - std_logit))

        return LogitKalmanState(
            x_prob=x_prob,
            x_logit=x_post,
            P_logit=P_post,
            K=K,
            innovation_logit=innovation,
            z_prob=z_prob,
            z_logit=z_logit,
            upper_95=upper_95,
            lower_95=lower_95,
        )

    def filter(self, observations: np.ndarray,
               timestamps: np.ndarray | None = None) -> LogitKalmanResult:
        """Run the logit-space filter over an observation sequence.

        Parameters
        ----------
        observations : np.ndarray
            Array of observed prices in [0, 1].
        timestamps : np.ndarray or None
            Optional timestamps.

        Returns
        -------
        LogitKalmanResult
            Complete filter output in both probability and logit space.
        """
        n = len(observations)
        states_prob = np.zeros(n)
        states_logit = np.zeros(n)
        covariances_logit = np.zeros(n)
        gains = np.zeros(n)
        innovations_logit = np.zeros(n)
        upper = np.zeros(n)
        lower = np.zeros(n)
        innovation_covs = np.zeros(n)

        for t in range(n):
            state = self.step(observations[t])
            states_prob[t] = state.x_prob
            states_logit[t] = state.x_logit
            covariances_logit[t] = state.P_logit
            gains[t] = state.K
            innovations_logit[t] = state.innovation_logit
            upper[t] = state.upper_95
            lower[t] = state.lower_95
            # S = P_prior + R_logit, stored implicitly
            innovation_covs[t] = state.P_logit / (state.K + EPSILON) if state.K > EPSILON else 1.0

        logger.info(
            "Logit filter: {} obs, final p={:.4f} [{:.4f}, {:.4f}]",
            n, states_prob[-1], lower[-1], upper[-1],
        )

        return LogitKalmanResult(
            states_prob=states_prob,
            states_logit=states_logit,
            covariances_logit=covariances_logit,
            gains=gains,
            innovations_logit=innovations_logit,
            observations_prob=observations.copy(),
            upper_95=upper,
            lower_95=lower,
            timestamps=timestamps,
            innovation_covariances=innovation_covs,
        )

    def reset(self, x0_prob: float | None = None, P0_logit: float = 1.0) -> None:
        """Reset the filter to initial conditions.

        Parameters
        ----------
        x0_prob : float or None
            New initial probability.
        P0_logit : float
            New initial covariance in logit space.
        """
        if x0_prob is not None:
            self.x_logit = float(logit(x0_prob))
        else:
            self.x_logit = 0.0
        self.P_logit = P0_logit
        self._initialized = x0_prob is not None
        self._step_count = 0
