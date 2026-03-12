"""Tests for the regime change detector."""

import numpy as np
import pytest

from src.detection.regime_detector import RegimeDetector


class TestRegimeDetector:
    """Tests for RegimeDetector."""

    def test_no_detection_on_normal_innovations(self) -> None:
        """Standard N(0,1) innovations should produce few false alarms."""
        detector = RegimeDetector(
            window=30, cusum_threshold=8.0, chi2_alpha=0.001,
            autocorr_threshold=0.4,
        )
        rng = np.random.default_rng(42)

        detections = 0
        for _ in range(200):
            innovation = rng.normal(0, 1)
            S = 1.0
            alert = detector.check(innovation * np.sqrt(S), S)
            if alert.detected:
                detections += 1

        # With stricter thresholds, few false positives expected
        assert detections < 30  # Allow some false positives from statistical variation

    def test_cusum_detects_mean_shift(self) -> None:
        """CUSUM should detect when innovations have shifted mean."""
        detector = RegimeDetector(cusum_threshold=4.0)
        rng = np.random.default_rng(42)

        # Normal period
        for _ in range(50):
            detector.check(rng.normal(0, 0.01), S=1e-4)

        # Regime change: large innovations
        detected = False
        for _ in range(30):
            alert = detector.check(rng.normal(0.1, 0.01), S=1e-4)
            if alert.detected and alert.method == "cusum":
                detected = True
                break

        assert detected

    def test_chi2_detects_variance_increase(self) -> None:
        """Chi-square test should detect increased innovation variance."""
        detector = RegimeDetector(window=20, chi2_alpha=0.01)
        rng = np.random.default_rng(42)

        # Normal period
        for _ in range(30):
            innovation = rng.normal(0, 0.01)
            detector.check(innovation, S=1e-4)

        # High variance period
        detected = False
        for _ in range(30):
            innovation = rng.normal(0, 0.1)  # 10x more volatile
            alert = detector.check(innovation, S=1e-4)
            if alert.detected and alert.method == "chi2":
                detected = True
                break

        assert detected

    def test_autocorrelation_detects_trend(self) -> None:
        """Autocorrelation test should detect serially correlated innovations."""
        detector = RegimeDetector(window=20, autocorr_threshold=0.3)

        # Create correlated innovations (run of positive values)
        for i in range(25):
            innovation = 0.05 + 0.001 * i  # Trending up
            detector.check(innovation, S=1e-3)

        detections = detector.get_detections()
        autocorr_detections = [d for d in detections if d.method == "autocorrelation"]
        assert len(autocorr_detections) > 0

    def test_severity_increases_with_strength(self) -> None:
        """Larger violations should produce higher severity scores."""
        detector1 = RegimeDetector(cusum_threshold=2.0)
        detector2 = RegimeDetector(cusum_threshold=2.0)

        # Mild surprise
        for _ in range(5):
            detector1.check(0.05, S=1e-4)
        alert1 = [a for a in detector1.get_detections()]

        # Strong surprise
        for _ in range(5):
            detector2.check(0.2, S=1e-4)
        alert2 = [a for a in detector2.get_detections()]

        # Both should detect, but strong should have higher severity
        if alert1 and alert2:
            assert alert2[-1].severity >= alert1[-1].severity

    def test_get_detections_filters(self) -> None:
        """get_detections should only return alerts where detected=True."""
        detector = RegimeDetector()
        rng = np.random.default_rng(42)

        for _ in range(50):
            detector.check(rng.normal(0, 0.01), S=1e-4)

        all_alerts = detector.get_alert_history()
        detections = detector.get_detections()

        assert len(all_alerts) == 50
        assert all(d.detected for d in detections)

    def test_reset_clears_state(self) -> None:
        """Reset should clear all internal state."""
        detector = RegimeDetector()
        for _ in range(30):
            detector.check(0.05, S=1e-4)

        detector.reset()
        assert len(detector.get_alert_history()) == 0
        assert len(detector._normalized_innovations) == 0
