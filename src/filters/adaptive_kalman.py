"""Adaptive Kalman filter with innovation-based process noise adjustment.

Extends the scalar Kalman filter to dynamically inflate the process noise Q
when the innovation sequence indicates a regime change (surprise event).
Also supports time-varying observation noise R_t.

When normalized innovations exceed a threshold, Q is temporarily inflated
to let the filter "catch up" to the new regime. The inflation decays
exponentially back to the baseline over subsequent steps.

References
----------
Mehra, R.K. (1972). "Approaches to Adaptive Filtering."
    IEEE Transactions on Automatic Control, 17(5), 693-698.
Mohamed, A.H. & Schwarz, K.P. (1999). "Adaptive Kalman Filtering for INS/GPS."
    Journal of Geodesy, 73(4), 193-203.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.filters.scalar_kalman import KalmanResult, KalmanState, ScalarKalmanFilter
from src.utils.math_helpers import DEFAULT_INITIAL_COVARIANCE, EPSILON


# Default threshold for normalized innovation to trigger Q inflation
DEFAULT_INNOVATION_THRESHOLD: float = 2.5

# Default factor by which Q is inflated during a detected regime change
DEFAULT_INFLATION_FACTOR: float = 10.0

# Default exponential decay rate for Q inflation (0 < decay < 1)
# After each step, inflation_factor *= decay
DEFAULT_DECAY_RATE: float = 0.8


@dataclass
class AdaptiveKalmanState(KalmanState):
    """Extended Kalman state with adaptive filter diagnostics.

    Parameters
    ----------
    Q_effective : float
        Effective process noise used in this step (may be inflated).
    R_effective : float
        Effective observation noise used in this step (may be dynamic).
    normalized_innovation : float
        Innovation divided by sqrt(innovation covariance).
    inflation_active : bool
        Whether Q inflation was active during this step.
    """

    Q_effective: float = 0.0
    R_effective: float = 0.0
    normalized_innovation: float = 0.0
    inflation_active: bool = False


class AdaptiveKalmanFilter:
    """Kalman filter with adaptive Q based on innovation monitoring.

    When the normalized innovation (y_t / sqrt(S_t)) exceeds a threshold,
    Q is temporarily inflated to allow the filter to track sudden changes.
    The inflation decays exponentially back to baseline.

    Also supports dynamic observation noise R_t passed per step.

    Parameters
    ----------
    Q_base : float
        Baseline process noise variance.
    R : float
        Default measurement noise variance (used when R_t not provided).
    threshold : float
        Normalized innovation threshold to trigger Q inflation.
    inflation : float
        Factor by which Q is multiplied during regime change.
    decay : float
        Exponential decay rate for inflation (0 < decay < 1).
    x0 : float or None
        Initial state estimate.
    P0 : float
        Initial state covariance.

    Examples
    --------
    >>> akf = AdaptiveKalmanFilter(Q_base=1e-4, R=1e-3)
    >>> state = akf.step(0.55)
    >>> print(f"Q_eff: {state.Q_effective:.2e}")
    """

    def __init__(
        self,
        Q_base: float,
        R: float,
        threshold: float = DEFAULT_INNOVATION_THRESHOLD,
        inflation: float = DEFAULT_INFLATION_FACTOR,
        decay: float = DEFAULT_DECAY_RATE,
        x0: float | None = None,
        P0: float = DEFAULT_INITIAL_COVARIANCE,
    ) -> None:
        if Q_base < 0:
            raise ValueError(f"Q_base must be non-negative, got {Q_base}")
        if R < 0:
            raise ValueError(f"R must be non-negative, got {R}")
        if not 0 < decay < 1:
            raise ValueError(f"Decay must be in (0, 1), got {decay}")

        self.Q_base = Q_base
        self.R_default = R
        self.threshold = threshold
        self.inflation_factor = inflation
        self.decay = decay

        self.x: float = x0 if x0 is not None else 0.0
        self.P: float = P0
        self._initialized: bool = x0 is not None
        self._step_count: int = 0

        # Current inflation multiplier (1.0 = no inflation)
        self._current_inflation: float = 1.0

        # History of effective Q values for diagnostics
        self._Q_history: list[float] = []
        self._R_history: list[float] = []

        logger.debug(
            "AdaptiveKalmanFilter: Q_base={:.2e}, R={:.2e}, threshold={:.1f}",
            Q_base, R, threshold,
        )

    @property
    def Q_effective(self) -> float:
        """Current effective process noise (Q_base * inflation multiplier)."""
        return self.Q_base * self._current_inflation

    def step(self, z: float, R_t: float | None = None) -> AdaptiveKalmanState:
        """Execute one predict-update cycle with adaptive noise.

        Parameters
        ----------
        z : float
            New observation (market price).
        R_t : float or None
            Time-varying observation noise. If None, uses default R.

        Returns
        -------
        AdaptiveKalmanState
            Complete filter state with adaptive diagnostics.
        """
        self._step_count += 1
        R = R_t if R_t is not None else self.R_default

        # Initialize on first observation
        if not self._initialized:
            self.x = z
            self._initialized = True

        # Compute effective Q with current inflation
        Q_eff = self.Q_effective

        # Predict
        x_prior = self.x
        P_prior = self.P + Q_eff

        # Innovation
        innovation = z - x_prior
        S = P_prior + R

        # Normalized innovation
        normalized = innovation / (np.sqrt(S) + EPSILON)

        # Check for regime change
        inflation_active = False
        if abs(normalized) > self.threshold:
            self._current_inflation = self.inflation_factor
            inflation_active = True
            logger.debug(
                "Regime detected at step {}: |ỹ|={:.2f} > {:.1f}, inflating Q by {:.0f}x",
                self._step_count, abs(normalized), self.threshold, self.inflation_factor,
            )
            # Recompute predict with inflated Q
            Q_eff = self.Q_effective
            P_prior = self.P + Q_eff
            S = P_prior + R
        else:
            # Decay inflation back to baseline
            if self._current_inflation > 1.0:
                self._current_inflation = max(
                    1.0,
                    1.0 + (self._current_inflation - 1.0) * self.decay,
                )

        # Kalman gain
        K = P_prior / (S + EPSILON)

        # Update
        x_post = x_prior + K * innovation
        P_post = (1.0 - K) * P_prior
        P_post = max(P_post, EPSILON)

        # Store state
        self.x = x_post
        self.P = P_post
        self._Q_history.append(Q_eff)
        self._R_history.append(R)

        return AdaptiveKalmanState(
            x=x_post,
            P=P_post,
            K=K,
            innovation=innovation,
            z=z,
            x_prior=x_prior,
            P_prior=P_prior,
            S=S,
            Q_effective=Q_eff,
            R_effective=R,
            normalized_innovation=float(normalized),
            inflation_active=inflation_active,
        )

    def filter(self, observations: np.ndarray,
               R_t_array: np.ndarray | None = None,
               timestamps: np.ndarray | None = None) -> KalmanResult:
        """Run the adaptive filter over an entire observation sequence.

        Parameters
        ----------
        observations : np.ndarray
            Array of observed prices.
        R_t_array : np.ndarray or None
            Optional array of time-varying observation noise values.
        timestamps : np.ndarray or None
            Optional timestamps.

        Returns
        -------
        KalmanResult
            Filter output with states, covariances, gains, and innovations.
        """
        n = len(observations)
        states = np.zeros(n)
        covariances = np.zeros(n)
        gains = np.zeros(n)
        innovations = np.zeros(n)
        innovation_covs = np.zeros(n)
        predicted_states = np.zeros(n)

        for t in range(n):
            R_t = float(R_t_array[t]) if R_t_array is not None else None
            state = self.step(observations[t], R_t=R_t)
            states[t] = state.x
            covariances[t] = state.P
            gains[t] = state.K
            innovations[t] = state.innovation
            innovation_covs[t] = state.S
            predicted_states[t] = state.x_prior

        logger.info(
            "Adaptive filter: {} obs, final x={:.4f}, Q inflations={}",
            n, self.x,
            sum(1 for q in self._Q_history if q > self.Q_base * 1.1),
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

    def get_Q_history(self) -> np.ndarray:
        """Return the history of effective Q values.

        Returns
        -------
        np.ndarray
            Array of Q_effective at each time step.
        """
        return np.array(self._Q_history)

    def get_R_history(self) -> np.ndarray:
        """Return the history of effective R values.

        Returns
        -------
        np.ndarray
            Array of R values used at each time step.
        """
        return np.array(self._R_history)

    def reset(self, x0: float | None = None,
              P0: float = DEFAULT_INITIAL_COVARIANCE) -> None:
        """Reset the filter to initial conditions.

        Parameters
        ----------
        x0 : float or None
            New initial state.
        P0 : float
            New initial covariance.
        """
        self.x = x0 if x0 is not None else 0.0
        self.P = P0
        self._initialized = x0 is not None
        self._step_count = 0
        self._current_inflation = 1.0
        self._Q_history.clear()
        self._R_history.clear()
