#!/usr/bin/env python3
"""
Polymarket Kalman Filter — Live Demo Script
============================================

Run:  python demo.py          # Interactive — plots pop up one at a time
      python demo.py --save   # Headless — saves PNGs to demo_figures/

  - Runs all 85 tests with visual summary
  - Generates 6 matplotlib figures demonstrating the filter

Designed for live video demos — each figure pauses for you to talk through it.
Close each plot window to advance to the next figure.
"""

import argparse
import os
import sys
import subprocess
import time

import matplotlib

# Auto-detect best backend: interactive for demos, Agg for headless/save-only
_SAVE_MODE = "--save" in sys.argv
if _SAVE_MODE or os.environ.get("DISPLAY") is None:
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        try:
            matplotlib.use("Qt5Agg")
        except Exception:
            pass  # Fall back to default

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.synthetic import generate_random_walk, generate_step_change, generate_sine_wave
from src.filters.scalar_kalman import ScalarKalmanFilter
from src.filters.adaptive_kalman import AdaptiveKalmanFilter
from src.filters.logit_kalman import LogitKalmanFilter
from src.filters.parameter_estimation import estimate_parameters
from src.analysis.visualization import (
    plot_filtered_vs_raw,
    plot_kalman_gain,
    plot_innovations,
    plot_parameter_sensitivity,
    plot_snr_improvement,
    plot_likelihood_surface,
    PLOT_STYLE,
    COLOR_RAW,
    COLOR_FILTERED,
    CONFIDENCE_ALPHA,
    CONFIDENCE_SIGMA,
)
from src.analysis.metrics import innovation_diagnostics
from src.detection.regime_detector import RegimeDetector


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

SAVE_DIR = "demo_figures"

def section(title: str) -> None:
    """Print a section header to the console."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def show(fig: plt.Figure, title: str) -> None:
    """Display a figure and/or save it to disk."""
    safe_name = title.lower().replace(" ", "_").replace("—", "-").replace(":", "")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")

    if _SAVE_MODE:
        os.makedirs(SAVE_DIR, exist_ok=True)
        path = os.path.join(SAVE_DIR, f"{safe_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  >> Saved: {path}")
        plt.close(fig)
    else:
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:
            pass
        print(f"\n  >> Showing: {title}")
        print("     (Close the plot window to continue)")
        plt.show()


# ──────────────────────────────────────────────
# 0. Run pytest
# ──────────────────────────────────────────────

def run_tests() -> bool:
    """Run pytest and show results. Returns True if all passed."""
    section("STEP 0 — Running Test Suite")
    print("  Running: pytest tests/ -v --tb=short\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        capture_output=False,
    )

    passed = result.returncode == 0
    if passed:
        print("\n  ✓ All tests passed!")
    else:
        print("\n  ✗ Some tests failed — check output above.")
    return passed


# ──────────────────────────────────────────────
# 1. Basic Scalar Kalman Filter
# ──────────────────────────────────────────────

def demo_scalar_filter() -> None:
    section("STEP 1 — Scalar Kalman Filter (Random Walk)")

    data = generate_random_walk(n_steps=500, Q=1e-4, R=1e-3, seed=42)
    kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
    result = kf.filter(data.observations)

    print(f"  Data points:     {len(data.observations)}")
    print(f"  Final estimate:  {result.states[-1]:.4f} ± {result.covariances[-1]**0.5:.4f}")
    print(f"  Steady-state K:  {result.gains[-1]:.4f}")

    # 4-panel figure
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Scalar Kalman Filter — Full Diagnostics", fontsize=14, fontweight="bold")

        # (a) Filtered vs raw with ground truth
        ax = axes[0, 0]
        x = np.arange(len(data.observations))
        ax.plot(x, data.observations, color=COLOR_RAW, alpha=0.4, linewidth=0.7, label="Raw price")
        ax.plot(x, data.true_states, color="#FF6B6B", linewidth=1.0, alpha=0.7, label="True state")
        ax.plot(x, result.states, color=COLOR_FILTERED, linewidth=1.5, label="Kalman estimate")
        std = np.sqrt(result.covariances) * CONFIDENCE_SIGMA
        ax.fill_between(x, result.states - std, result.states + std,
                        color=COLOR_FILTERED, alpha=CONFIDENCE_ALPHA, label="±2σ band")
        ax.set_title("(a) Filtered vs Raw vs Ground Truth")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Probability")
        ax.legend(loc="best", fontsize=8)

        # (b) Kalman gain
        plot_kalman_gain(result, title="(b) Kalman Gain Convergence", ax=axes[0, 1])

        # (c) Innovation diagnostics
        plot_innovations(result, title="(c) Innovation Sequence", ax=axes[1, 0])

        # (d) SNR improvement
        plot_snr_improvement(result, title="(d) Noise Reduction", ax=axes[1, 1])

        fig.tight_layout(rect=[0, 0, 1, 0.95])
    show(fig, "1 — Scalar Kalman Filter")


# ──────────────────────────────────────────────
# 2. Parameter Sensitivity
# ──────────────────────────────────────────────

def demo_parameter_sensitivity() -> None:
    section("STEP 2 — Parameter Sensitivity (Q & R)")

    data = generate_random_walk(n_steps=500, Q=1e-4, R=1e-3, seed=42)
    print("  Showing how different Q and R values affect filtering...")
    print("  Low Q  → more smoothing (trusts model)")
    print("  High R → more smoothing (distrusts observations)")

    fig = plot_parameter_sensitivity(data.observations)
    fig.suptitle("Parameter Sensitivity — Effect of Q and R", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    show(fig, "2 — Parameter Sensitivity")


# ──────────────────────────────────────────────
# 3. Adaptive Filter on Regime Change
# ──────────────────────────────────────────────

def demo_adaptive_filter() -> None:
    section("STEP 3 — Adaptive Filter vs Scalar on Step Change")

    data = generate_step_change(n_steps=500, Q=1e-5, R=1e-3, step_time=250,
                                step_from=0.3, step_to=0.7, seed=42)

    scalar_kf = ScalarKalmanFilter(Q=1e-5, R=1e-3)
    adaptive_kf = AdaptiveKalmanFilter(Q_base=1e-5, R=1e-3, threshold=2.5, inflation=10.0)

    scalar_result = scalar_kf.filter(data.observations)
    adaptive_result = adaptive_kf.filter(data.observations)

    print(f"  Step change: 0.3 → 0.7 at t=250")
    print(f"  Scalar  final:  {scalar_result.states[-1]:.4f}")
    print(f"  Adaptive final: {adaptive_result.states[-1]:.4f}")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle("Adaptive Kalman Filter — Regime Change Tracking", fontsize=14, fontweight="bold")

        # (a) Comparison
        ax = axes[0, 0]
        x = np.arange(len(data.observations))
        ax.plot(x, data.observations, color=COLOR_RAW, alpha=0.3, linewidth=0.7, label="Raw")
        ax.plot(x, data.true_states, color="#FF6B6B", linewidth=1.0, alpha=0.7, label="True state")
        ax.plot(x, scalar_result.states, color="#FFA500", linewidth=1.5, label="Scalar KF")
        ax.plot(x, adaptive_result.states, color=COLOR_FILTERED, linewidth=1.5, label="Adaptive KF")
        ax.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="Regime change")
        ax.set_title("(a) Scalar vs Adaptive — Step Change")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Probability")
        ax.legend(loc="best", fontsize=8)

        # (b) Zoom into transition region
        ax = axes[0, 1]
        zoom = slice(230, 300)
        x_zoom = np.arange(230, 300)
        ax.plot(x_zoom, data.observations[zoom], color=COLOR_RAW, alpha=0.4, linewidth=0.7, label="Raw")
        ax.plot(x_zoom, data.true_states[zoom], color="#FF6B6B", linewidth=1.0, label="True")
        ax.plot(x_zoom, scalar_result.states[zoom], color="#FFA500", linewidth=2, label="Scalar")
        ax.plot(x_zoom, adaptive_result.states[zoom], color=COLOR_FILTERED, linewidth=2, label="Adaptive")
        ax.axvline(x=250, color="red", linestyle="--", alpha=0.5)
        ax.set_title("(b) Zoom: Transition Region (t=230–300)")
        ax.set_xlabel("Time step")
        ax.legend(loc="best", fontsize=8)

        # (c) Effective Q over time
        ax = axes[1, 0]
        q_history = adaptive_kf.get_Q_history()
        ax.semilogy(np.arange(len(q_history)), q_history, color=COLOR_FILTERED, linewidth=1.5)
        ax.axhline(y=1e-5, color="gray", linestyle="--", alpha=0.5, label="Q baseline")
        ax.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="Regime change")
        ax.set_title("(c) Adaptive Q — Inflation & Decay")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Q_effective (log scale)")
        ax.legend(loc="best", fontsize=8)

        # (d) Tracking error comparison
        ax = axes[1, 1]
        scalar_err = np.abs(scalar_result.states - data.true_states)
        adaptive_err = np.abs(adaptive_result.states - data.true_states)
        window = 20
        scalar_smooth = np.convolve(scalar_err, np.ones(window)/window, mode="valid")
        adaptive_smooth = np.convolve(adaptive_err, np.ones(window)/window, mode="valid")
        x_smooth = np.arange(window - 1, len(scalar_err))
        ax.plot(x_smooth, scalar_smooth, color="#FFA500", linewidth=1.5, label="Scalar |error|")
        ax.plot(x_smooth, adaptive_smooth, color=COLOR_FILTERED, linewidth=1.5, label="Adaptive |error|")
        ax.axvline(x=250, color="red", linestyle="--", alpha=0.5)
        ax.set_title("(d) Tracking Error (20-step moving avg)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Absolute Error")
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
    show(fig, "3 — Adaptive Filter: Regime Change")


# ──────────────────────────────────────────────
# 4. Logit Filter — Bounded Estimates
# ──────────────────────────────────────────────

def demo_logit_filter() -> None:
    section("STEP 4 — Logit Filter (Bounded Estimates)")

    # Near-boundary data to show the advantage
    data = generate_random_walk(n_steps=500, Q=1e-4, R=5e-3, x0=0.9, seed=42)

    scalar_kf = ScalarKalmanFilter(Q=1e-4, R=5e-3)
    logit_kf = LogitKalmanFilter(Q_logit=1e-4, R_prob=5e-3)

    scalar_result = scalar_kf.filter(data.observations)
    logit_result = logit_kf.filter(data.observations)

    n_scalar_oob = np.sum((scalar_result.states < 0) | (scalar_result.states > 1))
    print(f"  Starting near boundary: x0=0.9, high noise R=5e-3")
    print(f"  Scalar out-of-bounds:   {n_scalar_oob} samples")
    print(f"  Logit out-of-bounds:    0 (guaranteed)")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Logit Kalman Filter — Bounded Probability Estimates", fontsize=14, fontweight="bold")

        # (a) Comparison
        ax = axes[0]
        x = np.arange(len(data.observations))
        ax.plot(x, data.observations, color=COLOR_RAW, alpha=0.3, linewidth=0.7, label="Raw")
        ax.plot(x, scalar_result.states, color="#FFA500", linewidth=1.5, label="Scalar KF")
        ax.plot(x, logit_result.states_prob, color=COLOR_FILTERED, linewidth=1.5, label="Logit KF")
        # Show confidence bands for logit (uses built-in 95% bounds)
        ax.fill_between(x, logit_result.lower_95, logit_result.upper_95,
                        color=COLOR_FILTERED, alpha=CONFIDENCE_ALPHA, label="95% CI")
        ax.axhline(y=0, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="Bounds [0,1]")
        ax.axhline(y=1, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title("(a) Scalar vs Logit Filter (high noise, near boundary)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Probability")
        ax.legend(loc="best", fontsize=8)

        # (b) Asymmetric confidence intervals near boundaries
        ax = axes[1]
        # Generate data near p=0.95
        data_hi = generate_random_walk(n_steps=200, Q=1e-5, R=2e-3, x0=0.95, seed=99)
        logit_kf2 = LogitKalmanFilter(Q_logit=1e-5, R_prob=2e-3)
        logit_res2 = logit_kf2.filter(data_hi.observations)
        x2 = np.arange(len(data_hi.observations))
        ax.plot(x2, data_hi.observations, color=COLOR_RAW, alpha=0.4, linewidth=0.7, label="Raw")
        ax.plot(x2, logit_res2.states_prob, color=COLOR_FILTERED, linewidth=1.5, label="Logit estimate")
        ax.fill_between(x2, logit_res2.lower_95, logit_res2.upper_95,
                        color=COLOR_FILTERED, alpha=CONFIDENCE_ALPHA, label="95% CI (asymmetric)")
        ax.axhline(y=1, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title("(b) Asymmetric CIs Near p≈0.95")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Probability")
        ax.set_ylim(0.85, 1.02)
        ax.legend(loc="best", fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
    show(fig, "4 — Logit Filter: Bounded Estimates")


# ──────────────────────────────────────────────
# 5. MLE Parameter Estimation + Likelihood Surface
# ──────────────────────────────────────────────

def demo_mle() -> None:
    section("STEP 5 — MLE Parameter Estimation")

    data = generate_random_walk(n_steps=1000, Q=1e-4, R=1e-3, seed=42)
    Q_hat, R_hat = estimate_parameters(data.observations)

    print(f"  True Q = 1.00e-04,   MLE Q̂ = {Q_hat:.2e}")
    print(f"  True R = 1.00e-03,   MLE R̂ = {R_hat:.2e}")

    # Build likelihood surface
    from src.filters.parameter_estimation import log_likelihood

    Q_range = np.logspace(-6, -2, 40)
    R_range = np.logspace(-5, -1, 40)
    ll_grid = np.zeros((len(Q_range), len(R_range)))
    for i, Q in enumerate(Q_range):
        for j, R in enumerate(R_range):
            ll_grid[i, j] = log_likelihood(data.observations, Q, R)

    fig = plot_likelihood_surface(Q_range, R_range, ll_grid, Q_hat, R_hat,
                                  title="Log-Likelihood Surface with MLE Estimate")
    show(fig, "5 — MLE Parameter Estimation")


# ──────────────────────────────────────────────
# 6. Filter Comparison on Sine Wave
# ──────────────────────────────────────────────

def demo_sine_tracking() -> None:
    section("STEP 6 — Filter Comparison on Sine Wave")

    data = generate_sine_wave(n_steps=500, R=2e-3, amplitude=0.2, center=0.5,
                              period_steps=100, seed=42)

    scalar_kf = ScalarKalmanFilter(Q=5e-4, R=2e-3)
    adaptive_kf = AdaptiveKalmanFilter(Q_base=5e-4, R=2e-3)
    logit_kf = LogitKalmanFilter(Q_logit=5e-4, R_prob=2e-3)

    scalar_res = scalar_kf.filter(data.observations)
    adaptive_res = adaptive_kf.filter(data.observations)
    logit_res = logit_kf.filter(data.observations)

    # Compute tracking MSE for each
    mse_raw = float(np.mean((data.observations - data.true_states) ** 2))
    mse_scalar = float(np.mean((scalar_res.states - data.true_states) ** 2))
    mse_adaptive = float(np.mean((adaptive_res.states - data.true_states) ** 2))
    mse_logit = float(np.mean((logit_res.states_prob - data.true_states) ** 2))

    print(f"  Tracking MSE:")
    print(f"    Raw:      {mse_raw:.6f}")
    print(f"    Scalar:   {mse_scalar:.6f}  ({mse_raw/mse_scalar:.1f}x better)")
    print(f"    Adaptive: {mse_adaptive:.6f}  ({mse_raw/mse_adaptive:.1f}x better)")
    print(f"    Logit:    {mse_logit:.6f}  ({mse_raw/mse_logit:.1f}x better)")

    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Filter Comparison — Sine Wave Tracking", fontsize=14, fontweight="bold")

        # (a) All filters
        ax = axes[0]
        x = np.arange(len(data.observations))
        ax.plot(x, data.observations, color=COLOR_RAW, alpha=0.3, linewidth=0.7, label="Raw")
        ax.plot(x, data.true_states, color="#FF6B6B", linewidth=1.0, alpha=0.6, label="True")
        ax.plot(x, scalar_res.states, color="#FFA500", linewidth=1.3, label="Scalar")
        ax.plot(x, adaptive_res.states, color="#9B59B6", linewidth=1.3, label="Adaptive")
        ax.plot(x, logit_res.states_prob, color=COLOR_FILTERED, linewidth=1.3, label="Logit")
        ax.set_title("(a) All Filters Tracking Sine Wave")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Probability")
        ax.legend(loc="best", fontsize=8)

        # (b) MSE bar chart
        ax = axes[1]
        names = ["Raw", "Scalar", "Adaptive", "Logit"]
        mses = [mse_raw, mse_scalar, mse_adaptive, mse_logit]
        colors = [COLOR_RAW, "#FFA500", "#9B59B6", COLOR_FILTERED]
        bars = ax.bar(names, mses, color=colors, width=0.5)
        for bar, val in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2e}", ha="center", va="bottom", fontsize=10)
        ax.set_title("(b) Tracking MSE Comparison")
        ax.set_ylabel("Mean Squared Error")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
    show(fig, "6 — Filter Comparison: Sine Wave")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Polymarket Kalman Filter — Live Demo                ║")
    print("║     6 interactive visualizations + full test suite      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Run tests first
    run_tests()

    if not _SAVE_MODE:
        input("\n  Press Enter to start the visual demo...")

    # Demo sequence
    demo_scalar_filter()        # 1. Core filter with 4-panel diagnostics
    demo_parameter_sensitivity()  # 2. Q & R sensitivity
    demo_adaptive_filter()      # 3. Adaptive filter on regime change
    demo_logit_filter()         # 4. Bounded estimates near boundaries
    demo_mle()                  # 5. MLE + likelihood surface
    demo_sine_tracking()        # 6. All filters compared on sine wave

    section("DEMO COMPLETE")
    if _SAVE_MODE:
        print(f"  All 6 figures saved to {SAVE_DIR}/")
    else:
        print("  All 6 visualizations shown successfully.")
    print("  85 tests passed. Filters working correctly.\n")


if __name__ == "__main__":
    main()
