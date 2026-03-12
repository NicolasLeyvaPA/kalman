"""Regime change detection via innovation monitoring.

Monitors the Kalman filter's innovation sequence for signs that the
underlying model has changed (regime shift). Under correct model specification,
the normalized innovation sequence should be N(0,1) and white (uncorrelated).

Three detection methods:
1. CUSUM — cumulative sum of squared normalized innovations
2. Chi-square — sliding window test on innovation magnitudes
3. Autocorrelation — tests for serial correlation in innovations

References
----------
Page, E.S. (1954). "Continuous Inspection Schemes." Biometrika, 41, 100-115.
Mehra, R.K. (1972). "Approaches to Adaptive Filtering."
"""

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from loguru import logger
from scipy import stats

from src.utils.math_helpers import EPSILON


@dataclass
class RegimeAlert:
    """Alert emitted when a regime change is detected.

    Parameters
    ----------
    detected : bool
        Whether a regime change was detected.
    severity : float
        Severity score from 0 (no detection) to 1 (strong detection).
    method : str
        Detection method that triggered: "cusum", "chi2", or "autocorrelation".
    description : str
        Human-readable description of the detection.
    timestamp : datetime
        When the detection occurred.
    """

    detected: bool
    severity: float
    method: str
    description: str
    timestamp: datetime


# Default sliding window size for chi-square and autocorrelation tests
DEFAULT_WINDOW: int = 20

# Default CUSUM threshold for detection
DEFAULT_CUSUM_THRESHOLD: float = 4.0

# Default significance level for chi-square test
DEFAULT_CHI2_ALPHA: float = 0.01

# Default autocorrelation threshold (absolute value)
DEFAULT_AUTOCORR_THRESHOLD: float = 0.3


class RegimeDetector:
    """Monitors innovations for regime changes.

    Maintains a rolling window of normalized innovations and runs three
    detection tests at each step.

    Parameters
    ----------
    window : int
        Size of the sliding window for chi-square and autocorrelation tests.
    cusum_threshold : float
        CUSUM detection threshold. Higher = fewer false alarms.
    chi2_alpha : float
        Significance level for the chi-square test.
    autocorr_threshold : float
        Threshold for innovation autocorrelation detection.

    Examples
    --------
    >>> detector = RegimeDetector()
    >>> alert = detector.check(innovation=0.05, S=0.001)
    >>> print(alert.detected)
    """

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        cusum_threshold: float = DEFAULT_CUSUM_THRESHOLD,
        chi2_alpha: float = DEFAULT_CHI2_ALPHA,
        autocorr_threshold: float = DEFAULT_AUTOCORR_THRESHOLD,
    ) -> None:
        self.window = window
        self.cusum_threshold = cusum_threshold
        self.chi2_alpha = chi2_alpha
        self.autocorr_threshold = autocorr_threshold

        self._normalized_innovations: list[float] = []
        self._cusum: float = 0.0
        self._cusum_min: float = 0.0
        self._alert_history: list[RegimeAlert] = []

    def check(
        self,
        innovation: float,
        S: float,
        timestamp: datetime | None = None,
    ) -> RegimeAlert:
        """Check for regime change given a new innovation.

        Parameters
        ----------
        innovation : float
            Filter innovation y_t.
        S : float
            Innovation covariance S_t.
        timestamp : datetime or None
            Current timestamp.

        Returns
        -------
        RegimeAlert
            Detection result with severity and method.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Normalize the innovation
        normalized = innovation / (np.sqrt(max(S, EPSILON)))
        self._normalized_innovations.append(normalized)

        # Run all three tests
        cusum_alert = self._check_cusum(normalized, timestamp)
        chi2_alert = self._check_chi2(timestamp)
        autocorr_alert = self._check_autocorrelation(timestamp)

        # Return the most severe alert
        alerts = [cusum_alert, chi2_alert, autocorr_alert]
        alerts = [a for a in alerts if a is not None]

        if not alerts:
            alert = RegimeAlert(
                detected=False, severity=0.0, method="none",
                description="No regime change detected", timestamp=timestamp,
            )
        else:
            alert = max(alerts, key=lambda a: a.severity)

        self._alert_history.append(alert)
        return alert

    def _check_cusum(self, normalized: float,
                      timestamp: datetime) -> RegimeAlert | None:
        """CUSUM test on squared normalized innovations.

        The CUSUM statistic accumulates deviations from the expected
        squared innovation (which should be 1 under H0). When it
        exceeds the threshold, a regime change is flagged.

        Parameters
        ----------
        normalized : float
            Normalized innovation (should be N(0,1) under H0).
        timestamp : datetime
            Current time.

        Returns
        -------
        RegimeAlert or None
            Alert if CUSUM threshold exceeded.
        """
        # CUSUM on squared innovations minus expected value (1.0)
        self._cusum += normalized ** 2 - 1.0
        self._cusum_min = min(self._cusum_min, self._cusum)

        cusum_stat = self._cusum - self._cusum_min

        if cusum_stat > self.cusum_threshold:
            severity = min(1.0, cusum_stat / (2.0 * self.cusum_threshold))
            # Reset CUSUM after detection
            self._cusum = 0.0
            self._cusum_min = 0.0
            return RegimeAlert(
                detected=True,
                severity=severity,
                method="cusum",
                description=f"CUSUM statistic {cusum_stat:.2f} > {self.cusum_threshold}",
                timestamp=timestamp,
            )
        return None

    def _check_chi2(self, timestamp: datetime) -> RegimeAlert | None:
        """Sliding window chi-square test.

        Over the last W observations, the sum of squared normalized
        innovations should follow chi-square(W). If it's too large,
        the filter is consistently surprised.

        Parameters
        ----------
        timestamp : datetime
            Current time.

        Returns
        -------
        RegimeAlert or None
            Alert if chi-square test rejects H0.
        """
        if len(self._normalized_innovations) < self.window:
            return None

        recent = np.array(self._normalized_innovations[-self.window:])
        chi2_stat = float(np.sum(recent ** 2))

        # Critical value for chi-square distribution
        critical = stats.chi2.ppf(1.0 - self.chi2_alpha, df=self.window)

        if chi2_stat > critical:
            p_value = 1.0 - stats.chi2.cdf(chi2_stat, df=self.window)
            severity = min(1.0, (chi2_stat - critical) / critical)
            return RegimeAlert(
                detected=True,
                severity=severity,
                method="chi2",
                description=(
                    f"Chi2 stat {chi2_stat:.1f} > critical {critical:.1f} "
                    f"(p={p_value:.4f})"
                ),
                timestamp=timestamp,
            )
        return None

    def _check_autocorrelation(self, timestamp: datetime) -> RegimeAlert | None:
        """Test for serial correlation in innovations.

        Under correct model specification, innovations should be white.
        After a regime change, you'll see a run of same-sign innovations.

        Parameters
        ----------
        timestamp : datetime
            Current time.

        Returns
        -------
        RegimeAlert or None
            Alert if significant autocorrelation detected.
        """
        if len(self._normalized_innovations) < self.window + 1:
            return None

        recent = np.array(self._normalized_innovations[-(self.window + 1):])
        autocorr = float(np.corrcoef(recent[:-1], recent[1:])[0, 1])

        if abs(autocorr) > self.autocorr_threshold:
            severity = min(1.0, abs(autocorr) / (2.0 * self.autocorr_threshold))
            return RegimeAlert(
                detected=True,
                severity=severity,
                method="autocorrelation",
                description=f"Innovation autocorrelation {autocorr:.3f} exceeds threshold",
                timestamp=timestamp,
            )
        return None

    def get_alert_history(self) -> list[RegimeAlert]:
        """Return all alerts (detected and not).

        Returns
        -------
        list[RegimeAlert]
            Complete alert history.
        """
        return self._alert_history.copy()

    def get_detections(self) -> list[RegimeAlert]:
        """Return only alerts where a regime change was detected.

        Returns
        -------
        list[RegimeAlert]
            Alerts with detected=True.
        """
        return [a for a in self._alert_history if a.detected]

    def reset(self) -> None:
        """Reset detector state."""
        self._normalized_innovations.clear()
        self._cusum = 0.0
        self._cusum_min = 0.0
        self._alert_history.clear()
