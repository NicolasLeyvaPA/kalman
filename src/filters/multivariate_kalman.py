"""Multivariate Kalman filter for correlated prediction markets.

Tracks multiple correlated markets simultaneously. When one market moves,
the cross-correlation structure propagates the update to all related markets
— even before their own prices change. This is the core value of the
multivariate extension.

State-space model:
    State: x_t = [p1, p2, ..., pn]^T  (n market probabilities)
    Transition: x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)  (n×n covariance)
    Observation: z_t = H_t x_t + v_t,  v_t ~ N(0, R_t)

H_t is the observation matrix. If all markets are observed, H = I.
If some markets have stale quotes, their rows are removed from H,
and the filter naturally handles the partial observation.

References
----------
Kalman, R.E. (1960). "A New Approach to Linear Filtering and Prediction Problems."
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.utils.math_helpers import EPSILON, ensure_positive_definite, symmetrize


@dataclass
class MultivariateKalmanState:
    """State of the multivariate Kalman filter after one cycle.

    Parameters
    ----------
    x : np.ndarray
        State estimate vector (n,).
    P : np.ndarray
        State covariance matrix (n, n).
    K : np.ndarray
        Kalman gain matrix (n, m) where m = number of observed markets.
    innovation : np.ndarray
        Innovation vector (m,).
    z : np.ndarray
        Observation vector (m,).
    observed_mask : np.ndarray
        Boolean mask indicating which markets were observed.
    """

    x: np.ndarray
    P: np.ndarray
    K: np.ndarray
    innovation: np.ndarray
    z: np.ndarray
    observed_mask: np.ndarray


@dataclass
class MultivariateKalmanResult:
    """Results from running the multivariate filter.

    Parameters
    ----------
    states : np.ndarray
        Filtered states at each time step, shape (T, n).
    covariances : np.ndarray
        State covariances at each step, shape (T, n, n).
    innovations : list[np.ndarray]
        Innovation vectors (variable length if partial observations).
    observations : np.ndarray
        Raw observations, shape (T, n). NaN for unobserved markets.
    timestamps : np.ndarray or None
        Timestamps.
    """

    states: np.ndarray
    covariances: np.ndarray
    innovations: list[np.ndarray]
    observations: np.ndarray
    timestamps: np.ndarray | None = None


class MultivariateKalmanFilter:
    """Multivariate Kalman filter tracking n correlated markets.

    The key insight: if markets A and B are positively correlated and you
    observe a price increase in A, the filter immediately increases its
    estimate for B — even if B hasn't traded yet.

    Parameters
    ----------
    n : int
        Number of markets to track.
    Q : np.ndarray
        n×n process noise covariance matrix. Diagonal elements are individual
        market noise; off-diagonal elements capture co-movement.
    R : np.ndarray
        n×n observation noise covariance matrix (typically diagonal).
    x0 : np.ndarray or None
        Initial state vector (n,). If None, initialized from first observation.
    P0 : np.ndarray or None
        Initial covariance matrix (n, n). If None, uses 0.25 * I.

    Examples
    --------
    >>> Q = np.array([[1e-4, 5e-5], [5e-5, 1e-4]])
    >>> R = np.diag([1e-3, 1e-3])
    >>> mkf = MultivariateKalmanFilter(n=2, Q=Q, R=R)
    >>> state = mkf.step(np.array([0.5, 0.6]))
    """

    def __init__(
        self,
        n: int,
        Q: np.ndarray,
        R: np.ndarray,
        x0: np.ndarray | None = None,
        P0: np.ndarray | None = None,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if Q.shape != (n, n):
            raise ValueError(f"Q must be ({n},{n}), got {Q.shape}")
        if R.shape != (n, n):
            raise ValueError(f"R must be ({n},{n}), got {R.shape}")

        self.n = n
        self.Q = Q.copy()
        self.R = R.copy()

        if x0 is not None:
            self.x = x0.copy()
            self._initialized = True
        else:
            self.x = np.full(n, 0.5)
            self._initialized = False

        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = 0.25 * np.eye(n)

        self._step_count = 0
        logger.debug("MultivariateKalmanFilter: n={}, Q diag={}", n, np.diag(Q))

    def predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Prediction step: project state and covariance forward.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (x_prior, P_prior) — predicted state and covariance.

        Notes
        -----
        x_prior = x  (random walk)
        P_prior = P + Q
        """
        x_prior = self.x.copy()
        P_prior = self.P + self.Q
        return x_prior, P_prior

    def update(
        self,
        z: np.ndarray,
        observed_mask: np.ndarray | None = None,
    ) -> MultivariateKalmanState:
        """Update step: incorporate new observations.

        Handles partial observations by selecting only the observed rows
        of the observation matrix H.

        Parameters
        ----------
        z : np.ndarray
            Observation vector (n,). Unobserved entries are ignored
            (their values don't matter if mask is provided).
        observed_mask : np.ndarray or None
            Boolean array (n,). True = market observed. If None, all observed.

        Returns
        -------
        MultivariateKalmanState
            Complete state after the update.

        Notes
        -----
        With full observations (H = I):
            K = P_prior @ inv(P_prior + R)
            x = x_prior + K @ (z - x_prior)
            P = (I - K) @ P_prior

        With partial observations (H = selection matrix):
            S = H @ P_prior @ H^T + R_obs
            K = P_prior @ H^T @ inv(S)
            x = x_prior + K @ (z_obs - H @ x_prior)
            P = (I - K @ H) @ P_prior
        """
        self._step_count += 1

        if observed_mask is None:
            observed_mask = np.ones(self.n, dtype=bool)

        # Initialize from first observation if needed
        if not self._initialized:
            self.x[observed_mask] = z[observed_mask]
            self._initialized = True

        # Predict
        x_prior, P_prior = self.predict()

        # Select observed dimensions
        obs_idx = np.where(observed_mask)[0]
        m = len(obs_idx)

        if m == 0:
            # No observations: just predict
            self.x = x_prior
            self.P = P_prior
            return MultivariateKalmanState(
                x=x_prior.copy(),
                P=P_prior.copy(),
                K=np.zeros((self.n, 0)),
                innovation=np.array([]),
                z=z.copy(),
                observed_mask=observed_mask.copy(),
            )

        # Build observation matrix H (m × n selection matrix)
        H = np.zeros((m, self.n))
        for i, idx in enumerate(obs_idx):
            H[i, idx] = 1.0

        # Observed components
        z_obs = z[obs_idx]
        R_obs = self.R[np.ix_(obs_idx, obs_idx)]

        # Innovation
        innovation = z_obs - H @ x_prior

        # Innovation covariance
        S = H @ P_prior @ H.T + R_obs

        # Kalman gain: K = P_prior @ H^T @ S^{-1}
        # Use solve for numerical stability: S @ K^T = (P_prior @ H^T)^T
        try:
            K = P_prior @ H.T @ np.linalg.solve(S, np.eye(m))
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if S is singular
            logger.warning("Innovation covariance singular, using pseudoinverse")
            K = P_prior @ H.T @ np.linalg.pinv(S)

        # State update
        x_post = x_prior + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        IKH = np.eye(self.n) - K @ H
        P_post = IKH @ P_prior @ IKH.T + K @ R_obs @ K.T

        # Ensure symmetry and positive definiteness
        P_post = symmetrize(P_post)
        P_post = ensure_positive_definite(P_post)

        # Store
        self.x = x_post
        self.P = P_post

        return MultivariateKalmanState(
            x=x_post.copy(),
            P=P_post.copy(),
            K=K.copy(),
            innovation=innovation.copy(),
            z=z.copy(),
            observed_mask=observed_mask.copy(),
        )

    def step(self, z: np.ndarray,
             observed_mask: np.ndarray | None = None) -> MultivariateKalmanState:
        """Convenience method: equivalent to update().

        Parameters
        ----------
        z : np.ndarray
            Observation vector.
        observed_mask : np.ndarray or None
            Which markets are observed.

        Returns
        -------
        MultivariateKalmanState
            Updated state.
        """
        return self.update(z, observed_mask)

    def filter(
        self,
        observations: np.ndarray,
        observed_masks: np.ndarray | None = None,
        timestamps: np.ndarray | None = None,
    ) -> MultivariateKalmanResult:
        """Run the filter over a sequence of multivariate observations.

        Parameters
        ----------
        observations : np.ndarray
            Observation matrix of shape (T, n). Use NaN for unobserved entries.
        observed_masks : np.ndarray or None
            Boolean mask matrix (T, n). If None, derived from non-NaN entries.
        timestamps : np.ndarray or None
            Optional timestamps.

        Returns
        -------
        MultivariateKalmanResult
            Complete filter output.
        """
        T = observations.shape[0]
        states = np.zeros((T, self.n))
        covariances = np.zeros((T, self.n, self.n))
        innovations: list[np.ndarray] = []

        for t in range(T):
            z = observations[t].copy()

            if observed_masks is not None:
                mask = observed_masks[t]
            else:
                mask = ~np.isnan(z)

            # Replace NaN with 0 (ignored by mask anyway)
            z = np.nan_to_num(z, nan=0.0)

            state = self.update(z, observed_mask=mask)
            states[t] = state.x
            covariances[t] = state.P
            innovations.append(state.innovation)

        logger.info(
            "Multivariate filter: {} steps, {} markets, final x={}",
            T, self.n, np.round(self.x, 4),
        )

        return MultivariateKalmanResult(
            states=states,
            covariances=covariances,
            innovations=innovations,
            observations=observations.copy(),
            timestamps=timestamps,
        )

    def reset(self, x0: np.ndarray | None = None,
              P0: np.ndarray | None = None) -> None:
        """Reset the filter.

        Parameters
        ----------
        x0 : np.ndarray or None
            New initial state.
        P0 : np.ndarray or None
            New initial covariance.
        """
        if x0 is not None:
            self.x = x0.copy()
            self._initialized = True
        else:
            self.x = np.full(self.n, 0.5)
            self._initialized = False

        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = 0.25 * np.eye(self.n)
        self._step_count = 0
