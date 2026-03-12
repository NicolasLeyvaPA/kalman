"""Backtesting framework for filter accuracy evaluation.

Runs filter variants on historical data and computes performance metrics.
For resolved markets, the true outcome (0 or 1) is known, enabling
proper scoring via Brier score, log loss, and calibration analysis.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.analysis.metrics import brier_score, calibration_curve, log_loss
from src.filters.adaptive_kalman import AdaptiveKalmanFilter
from src.filters.logit_kalman import LogitKalmanFilter
from src.filters.scalar_kalman import ScalarKalmanFilter


@dataclass
class BacktestResult:
    """Results from a single backtest run.

    Parameters
    ----------
    filter_name : str
        Name of the filter variant.
    predictions : np.ndarray
        Filter output probabilities at each time step.
    brier : float
        Brier score (lower is better).
    logloss : float
        Log loss (lower is better).
    innovation_frac_2sigma : float
        Fraction of innovations within ±2σ.
    lead_time_steps : int
        Steps the filter's 0.5-crossing leads the raw price's crossing.
        Positive = filter leads.
    """

    filter_name: str
    predictions: np.ndarray
    brier: float
    logloss: float
    innovation_frac_2sigma: float
    lead_time_steps: int


class FilterBacktest:
    """Backtest framework comparing filter variants.

    Runs multiple filter types on the same historical data and
    computes standardized performance metrics.

    Parameters
    ----------
    Q : float
        Process noise for scalar and adaptive filters.
    R : float
        Observation noise for all filters.
    Q_logit : float
        Process noise in logit space for logit filter.

    Examples
    --------
    >>> bt = FilterBacktest()
    >>> results = bt.run(observations, outcome=1)
    """

    def __init__(
        self,
        Q: float = 1e-4,
        R: float = 1e-3,
        Q_logit: float = 1e-3,
    ) -> None:
        self.Q = Q
        self.R = R
        self.Q_logit = Q_logit

    def run(
        self,
        observations: np.ndarray,
        outcome: int,
    ) -> list[BacktestResult]:
        """Run all filter variants on a single market's data.

        Parameters
        ----------
        observations : np.ndarray
            Historical price observations.
        outcome : int
            True outcome (0 or 1) for scoring.

        Returns
        -------
        list[BacktestResult]
            Results for each filter variant: raw, SMA-20, scalar Kalman,
            adaptive Kalman, logit Kalman.
        """
        results: list[BacktestResult] = []
        outcomes = np.full(len(observations), outcome)

        # 1. Raw prices (baseline)
        results.append(self._evaluate(
            "Raw Price", observations, observations, outcomes,
        ))

        # 2. Simple Moving Average (20-period)
        sma = self._simple_moving_average(observations, window=20)
        results.append(self._evaluate("SMA-20", sma, observations, outcomes))

        # 3. Scalar Kalman Filter
        kf = ScalarKalmanFilter(Q=self.Q, R=self.R)
        kf_result = kf.filter(observations)
        results.append(self._evaluate(
            "Scalar Kalman", kf_result.states, observations, outcomes,
            innovations=kf_result.innovations,
            innovation_covs=kf_result.innovation_covariances,
        ))

        # 4. Adaptive Kalman Filter
        akf = AdaptiveKalmanFilter(Q_base=self.Q, R=self.R)
        akf_result = akf.filter(observations)
        results.append(self._evaluate(
            "Adaptive Kalman", akf_result.states, observations, outcomes,
            innovations=akf_result.innovations,
            innovation_covs=akf_result.innovation_covariances,
        ))

        # 5. Logit Kalman Filter
        lkf = LogitKalmanFilter(Q_logit=self.Q_logit, R_prob=self.R)
        lkf_result = lkf.filter(observations)
        results.append(self._evaluate(
            "Logit Kalman", lkf_result.states_prob, observations, outcomes,
        ))

        logger.info("Backtest complete: {} variants evaluated", len(results))
        return results

    def _evaluate(
        self,
        name: str,
        predictions: np.ndarray,
        raw_observations: np.ndarray,
        outcomes: np.ndarray,
        innovations: np.ndarray | None = None,
        innovation_covs: np.ndarray | None = None,
    ) -> BacktestResult:
        """Evaluate a single filter's predictions.

        Parameters
        ----------
        name : str
            Filter name.
        predictions : np.ndarray
            Filter output probabilities.
        raw_observations : np.ndarray
            Raw price observations (for lead time calculation).
        outcomes : np.ndarray
            Binary outcomes.
        innovations : np.ndarray or None
            Innovation sequence (if available).
        innovation_covs : np.ndarray or None
            Innovation covariances (if available).

        Returns
        -------
        BacktestResult
            Evaluation metrics.
        """
        # Clip predictions to valid probability range
        preds = np.clip(predictions, 1e-6, 1.0 - 1e-6)

        bs = brier_score(preds, outcomes)
        ll = log_loss(preds, outcomes)

        # Innovation diagnostics
        innov_frac = 0.0
        if innovations is not None and innovation_covs is not None:
            S = np.maximum(innovation_covs, 1e-15)
            normalized = innovations / np.sqrt(S)
            innov_frac = float(np.mean(np.abs(normalized[10:]) < 2.0))

        # Lead time: when does the filter cross 0.5 vs raw price?
        lead = self._compute_lead_time(preds, raw_observations)

        return BacktestResult(
            filter_name=name,
            predictions=preds,
            brier=bs,
            logloss=ll,
            innovation_frac_2sigma=innov_frac,
            lead_time_steps=lead,
        )

    @staticmethod
    def _simple_moving_average(data: np.ndarray, window: int = 20) -> np.ndarray:
        """Compute simple moving average with edge padding.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        window : int
            Window size.

        Returns
        -------
        np.ndarray
            Smoothed data (same length as input).
        """
        cumsum = np.cumsum(np.insert(data, 0, 0))
        sma = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window + 1)
            sma[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
        return sma

    @staticmethod
    def _compute_lead_time(
        filtered: np.ndarray,
        raw: np.ndarray,
    ) -> int:
        """Compute how many steps the filter's 0.5-crossing leads the raw.

        Parameters
        ----------
        filtered : np.ndarray
            Filtered probability estimates.
        raw : np.ndarray
            Raw price observations.

        Returns
        -------
        int
            Lead time in steps. Positive = filter leads.
        """
        # Find first crossing of 0.5 for each
        filtered_cross = _find_crossing(filtered, 0.5)
        raw_cross = _find_crossing(raw, 0.5)

        if filtered_cross is None or raw_cross is None:
            return 0

        return raw_cross - filtered_cross


def _find_crossing(series: np.ndarray, level: float) -> int | None:
    """Find the first time a series crosses a level.

    Parameters
    ----------
    series : np.ndarray
        Time series data.
    level : float
        Crossing level.

    Returns
    -------
    int or None
        Index of first crossing, or None if no crossing.
    """
    above = series > level
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]
    return int(crossings[0]) if len(crossings) > 0 else None
