"""Generate all figures for the academic paper.

Run from the project root:
    python paper/generate_figures.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger
logger.disable("src")

from src.data.synthetic import generate_random_walk, generate_step_change
from src.filters.scalar_kalman import ScalarKalmanFilter
from src.filters.adaptive_kalman import AdaptiveKalmanFilter
from src.filters.logit_kalman import LogitKalmanFilter
from src.filters.multivariate_kalman import MultivariateKalmanFilter
from src.filters.parameter_estimation import estimate_parameters, likelihood_surface
from src.detection.regime_detector import RegimeDetector
from src.pipeline.backtest import FilterBacktest
from src.analysis.metrics import brier_score, log_loss, calibration_curve
from src.analysis.visualization import (
    plot_filtered_vs_raw, plot_kalman_gain, plot_innovations,
    plot_parameter_sensitivity, plot_snr_improvement, plot_likelihood_surface,
)
from src.analysis.correlation import (
    estimate_cross_covariance, correlation_matrix, ledoit_wolf_shrinkage,
)
from src.utils.transforms import logit, sigmoid

sns.set_theme(style="whitegrid")

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  Saved {name}")


# ── Figure 1: Filtered vs Raw ──────────────────────────────────────
def fig_filtered_vs_raw():
    print("Generating: filtered vs raw...")
    data = generate_random_walk(n_steps=500, Q=1e-4, R=2e-3, x0=0.55, seed=42)
    kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
    result = kf.filter(data.observations)

    fig = plot_filtered_vs_raw(result, title="Kalman Filtered Probability")
    ax = fig.axes[0]
    ax.plot(data.true_states, color="green", linewidth=1.0, alpha=0.8, label="True state")
    ax.legend()
    save(fig, "fig_filtered_vs_raw.pdf")


# ── Figure 2: Kalman Gain ──────────────────────────────────────────
def fig_kalman_gain():
    print("Generating: Kalman gain...")
    data = generate_random_walk(n_steps=500, Q=1e-4, R=1e-3, seed=42)
    kf = ScalarKalmanFilter(Q=1e-4, R=1e-3)
    result = kf.filter(data.observations)
    fig = plot_kalman_gain(result, title="Kalman Gain Convergence")
    save(fig, "fig_kalman_gain.pdf")


# ── Figure 3: Parameter Sensitivity ───────────────────────────────
def fig_parameter_sensitivity():
    print("Generating: parameter sensitivity...")
    data = generate_random_walk(n_steps=500, Q=1e-4, R=2e-3, x0=0.55, seed=42)
    fig = plot_parameter_sensitivity(data.observations)
    save(fig, "fig_parameter_sensitivity.pdf")


# ── Figure 4: MLE Likelihood Surface ──────────────────────────────
def fig_likelihood_surface():
    print("Generating: likelihood surface...")
    data = generate_random_walk(n_steps=500, Q=1e-4, R=2e-3, x0=0.55, seed=42)
    Q_hat, R_hat = estimate_parameters(data.observations)
    Q_range = np.logspace(-7, -2, 25)
    R_range = np.logspace(-5, -1, 25)
    ll_grid = likelihood_surface(data.observations, Q_range, R_range)
    fig = plot_likelihood_surface(Q_range, R_range, ll_grid, Q_hat, R_hat)
    save(fig, "fig_likelihood_surface.pdf")
    print(f"  MLE: Q={Q_hat:.2e}, R={R_hat:.2e} (true Q=1e-4, R=2e-3)")


# ── Figure 5: Adaptive vs Basic ───────────────────────────────────
def fig_adaptive_comparison():
    print("Generating: adaptive vs basic...")
    data = generate_step_change(
        n_steps=500, step_time=250,
        step_from=0.3, step_to=0.75,
        Q=1e-6, R=1e-3, seed=42,
    )

    basic = ScalarKalmanFilter(Q=1e-5, R=1e-3)
    result_basic = basic.filter(data.observations)

    adaptive = AdaptiveKalmanFilter(
        Q_base=1e-5, R=1e-3,
        threshold=2.5, inflation=15.0, decay=0.8,
    )
    result_adaptive = adaptive.filter(data.observations)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(data.observations, color="gray", alpha=0.4, linewidth=0.8, label="Raw")
    ax1.plot(data.true_states, color="green", linewidth=1.0, alpha=0.7, label="True")
    ax1.plot(result_basic.states, color="#00CED1", linewidth=1.5, label="Basic Kalman")
    ax1.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="Regime change")
    ax1.set_title("Basic Scalar Kalman Filter")
    ax1.legend()
    ax1.set_ylabel("Probability")

    ax2.plot(data.observations, color="gray", alpha=0.4, linewidth=0.8, label="Raw")
    ax2.plot(data.true_states, color="green", linewidth=1.0, alpha=0.7, label="True")
    ax2.plot(result_adaptive.states, color="#FF6347", linewidth=1.5, label="Adaptive Kalman")
    ax2.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="Regime change")
    ax2.set_title("Adaptive Kalman Filter (innovation monitoring)")
    ax2.legend()
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Probability")

    plt.tight_layout()
    save(fig, "fig_adaptive_comparison.pdf")

    # Tracking error
    error_basic = np.abs(result_basic.states - data.true_states)
    error_adaptive = np.abs(result_adaptive.states - data.true_states)

    fig2, ax = plt.subplots(figsize=(12, 4))
    ax.plot(error_basic, color="#00CED1", alpha=0.7, label=f"Basic (MAE={error_basic.mean():.4f})")
    ax.plot(error_adaptive, color="#FF6347", alpha=0.7, label=f"Adaptive (MAE={error_adaptive.mean():.4f})")
    ax.axvline(x=250, color="red", linestyle="--", alpha=0.3)
    ax.set_xlabel("Time step")
    ax.set_ylabel("|Estimate - True|")
    ax.set_title("Absolute Tracking Error")
    ax.legend()
    plt.tight_layout()
    save(fig2, "fig_tracking_error.pdf")

    # Q inflation timeline
    Q_hist = adaptive.get_Q_history()
    fig3, ax = plt.subplots(figsize=(12, 3.5))
    ax.semilogy(Q_hist, color="#FF6347", linewidth=1.0)
    ax.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="Regime change")
    ax.axhline(y=1e-5, color="blue", linestyle=":", alpha=0.5, label="$Q_{base}$")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Effective $Q$ (log scale)")
    ax.set_title("Process Noise Inflation Timeline")
    ax.legend()
    plt.tight_layout()
    save(fig3, "fig_q_inflation.pdf")

    post_step = slice(250, 300)
    print(f"  Post-step MAE: Basic={error_basic[post_step].mean():.4f}, Adaptive={error_adaptive[post_step].mean():.4f}")


# ── Figure 6: Logit Bounded vs Unbounded ──────────────────────────
def fig_logit_bounded():
    print("Generating: logit bounded vs unbounded...")
    data = generate_random_walk(n_steps=300, Q=5e-4, R=5e-3, x0=0.92, seed=42)

    regular = ScalarKalmanFilter(Q=5e-4, R=5e-3)
    result_reg = regular.filter(data.observations)

    logit_f = LogitKalmanFilter(Q_logit=5e-3, R_prob=5e-3)
    result_logit = logit_f.filter(data.observations)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(data.observations, color="gray", alpha=0.4, linewidth=0.8, label="Raw")
    ax1.plot(result_reg.states, color="#00CED1", linewidth=1.5, label="Regular Kalman")
    std_reg = np.sqrt(result_reg.covariances) * 2
    ax1.fill_between(range(300), result_reg.states - std_reg, result_reg.states + std_reg,
                     color="#00CED1", alpha=0.15, label=r"$\pm 2\sigma$ CI")
    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="$y=1$ boundary")
    ax1.axhline(y=0.0, color="red", linestyle="--", alpha=0.5)
    exceeded = np.sum((result_reg.states + std_reg) > 1.0)
    ax1.set_title(f"Regular Filter: CI exceeds 1.0 at {exceeded} points")
    ax1.legend(fontsize=9)
    ax1.set_ylabel("Probability")

    ax2.plot(data.observations, color="gray", alpha=0.4, linewidth=0.8, label="Raw")
    ax2.plot(result_logit.states_prob, color="#FF6347", linewidth=1.5, label="Logit Kalman")
    ax2.fill_between(range(300), result_logit.lower_95, result_logit.upper_95,
                     color="#FF6347", alpha=0.15, label=r"$\pm 2\sigma$ CI (bounded)")
    ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
    ax2.axhline(y=0.0, color="red", linestyle="--", alpha=0.5)
    ax2.set_title("Logit Filter: CI always in [0, 1]")
    ax2.legend(fontsize=9)
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Probability")

    plt.tight_layout()
    save(fig, "fig_logit_bounded.pdf")


# ── Figure 7: Asymmetric CIs ─────────────────────────────────────
def fig_asymmetric_ci():
    print("Generating: asymmetric CIs...")
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))

    for ax, p0 in zip(axes, probs):
        data = generate_random_walk(n_steps=150, Q=1e-4, R=2e-3, x0=p0, seed=42)
        lkf = LogitKalmanFilter(Q_logit=1e-3, R_prob=2e-3)
        result = lkf.filter(data.observations)
        ax.plot(result.states_prob, color="#FF6347", linewidth=1.5)
        ax.fill_between(range(150), result.lower_95, result.upper_95,
                        color="#FF6347", alpha=0.15)
        ax.set_title(f"$p \\approx {p0}$")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0, color="gray", linewidth=0.5)
        ax.axhline(y=1, color="gray", linewidth=0.5)

    axes[0].set_ylabel("Probability")
    fig.suptitle("Logit Filter: Asymmetric Confidence Intervals at Different Price Levels", y=1.02)
    plt.tight_layout()
    save(fig, "fig_asymmetric_ci.pdf")


# ── Figure 8: Correlation Heatmap ────────────────────────────────
def fig_correlation_heatmap():
    print("Generating: correlation heatmap...")
    rng = np.random.default_rng(42)
    T, n = 500, 3
    market_names = ["Fed Pause", "Fed Cut", "Infl < 3%"]

    Q_true = np.array([
        [1e-4, 7e-5, -4e-5],
        [7e-5, 1e-4, -3e-5],
        [-4e-5, -3e-5, 8e-5],
    ])
    R_true = np.diag([1e-3, 1.5e-3, 2e-3])

    L_Q = np.linalg.cholesky(Q_true)
    true_states = np.zeros((T, n))
    true_states[0] = [0.55, 0.40, 0.60]
    for t in range(1, T):
        true_states[t] = np.clip(true_states[t-1] + L_Q @ rng.standard_normal(n), 0.01, 0.99)

    L_R = np.linalg.cholesky(R_true)
    observations = np.zeros((T, n))
    for t in range(T):
        observations[t] = np.clip(true_states[t] + L_R @ rng.standard_normal(n), 0.01, 0.99)

    prices = {name: observations[:, i] for i, name in enumerate(market_names)}
    Q_est, _ = estimate_cross_covariance(prices)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    true_corr = correlation_matrix(Q_true)
    sns.heatmap(true_corr, annot=True, fmt=".3f", ax=ax1,
                xticklabels=market_names, yticklabels=market_names,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    ax1.set_title("True Correlation")

    est_corr = correlation_matrix(Q_est)
    sns.heatmap(est_corr, annot=True, fmt=".3f", ax=ax2,
                xticklabels=market_names, yticklabels=market_names,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    ax2.set_title("Estimated (Ledoit-Wolf)")

    plt.tight_layout()
    save(fig, "fig_correlation_heatmap.pdf")

    # Multivariate vs Independent
    R_filter = np.diag([1e-3] * n)
    mkf = MultivariateKalmanFilter(n=n, Q=Q_est, R=R_filter)
    result_mv = mkf.filter(observations)

    result_indep = np.zeros((T, n))
    for j in range(n):
        skf = ScalarKalmanFilter(Q=Q_est[j, j], R=R_filter[j, j])
        r = skf.filter(observations[:, j])
        result_indep[:, j] = r.states

    colors = ["#00CED1", "#FF6347", "#9370DB"]
    fig2, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for j in range(n):
        ax = axes[j]
        ax.plot(observations[:, j], color="gray", alpha=0.3, linewidth=0.5, label="Raw")
        ax.plot(true_states[:, j], color="green", linewidth=0.8, alpha=0.6, label="True")
        ax.plot(result_mv.states[:, j], color=colors[j], linewidth=1.5, label="Multivariate")
        ax.plot(result_indep[:, j], color=colors[j], linewidth=1.0, linestyle="--",
                alpha=0.6, label="Independent")
        ax.set_ylabel("Probability")
        ax.set_title(market_names[j])
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time step")
    fig2.suptitle("Multivariate vs Independent Filters", y=1.01)
    plt.tight_layout()
    save(fig2, "fig_multivariate_vs_independent.pdf")

    # Print MSE
    print("  MSE comparison:")
    for j in range(n):
        mse_indep = np.mean((result_indep[:, j] - true_states[:, j]) ** 2)
        mse_mv = np.mean((result_mv.states[:, j] - true_states[:, j]) ** 2)
        improvement = (mse_indep - mse_mv) / mse_indep * 100
        print(f"    {market_names[j]}: indep={mse_indep:.2e}, mv={mse_mv:.2e}, improvement={improvement:.1f}%")


# ── Figure 9: Regime Detection ────────────────────────────────────
def fig_regime_detection():
    print("Generating: regime detection...")
    data = generate_step_change(
        n_steps=500, step_time=250,
        step_from=0.35, step_to=0.72,
        Q=1e-6, R=1e-3, seed=42,
    )

    kf = ScalarKalmanFilter(Q=1e-5, R=1e-3)
    result = kf.filter(data.observations)

    detector = RegimeDetector(window=20, cusum_threshold=4.0)
    alerts = []
    for t in range(len(data.observations)):
        alert = detector.check(
            innovation=result.innovations[t],
            S=result.innovation_covariances[t],
        )
        alerts.append(alert)

    detection_times = [t for t, a in enumerate(alerts) if a.detected]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax1.plot(data.observations, color="gray", alpha=0.4, linewidth=0.8, label="Raw")
    ax1.plot(data.true_states, color="green", linewidth=1.0, alpha=0.6, label="True")
    ax1.plot(result.states, color="#00CED1", linewidth=1.5, label="Filtered")
    ax1.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="True change")
    for t in detection_times:
        ax1.axvline(x=t, color="orange", alpha=0.3, linewidth=1)
    ax1.set_ylabel("Probability")
    ax1.set_title("Filtered Price with Regime Detection")
    ax1.legend(loc="best", fontsize=8)

    S = result.innovation_covariances
    normalized = result.innovations / np.sqrt(np.maximum(S, 1e-15))
    ax2.plot(normalized, color="#00CED1", alpha=0.7, linewidth=0.8)
    ax2.axhline(y=2, color="red", linestyle="--", alpha=0.3)
    ax2.axhline(y=-2, color="red", linestyle="--", alpha=0.3)
    ax2.axvline(x=250, color="red", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Normalized Innovation")
    ax2.set_title("Innovation Sequence")

    severities = [a.severity if a.detected else 0 for a in alerts]
    ax3.bar(range(len(severities)), severities, color="orange", width=1.0)
    ax3.axvline(x=250, color="red", linestyle="--", alpha=0.5, label="True change")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Severity")
    ax3.set_title("Regime Change Detections")
    ax3.legend()

    plt.tight_layout()
    save(fig, "fig_regime_detection.pdf")

    if detection_times:
        post = [t for t in detection_times if t >= 245]
        if post:
            print(f"  Detection latency: {min(post) - 250} steps")


# ── Figure 10: Backtest Brier Scores ─────────────────────────────
def fig_backtest():
    print("Generating: backtest comparison...")
    scenarios = [
        {"name": "Resolves YES (high)", "x0": 0.75, "Q": 1e-5, "R": 2e-3, "outcome": 1},
        {"name": "Resolves NO (low)", "x0": 0.25, "Q": 1e-5, "R": 2e-3, "outcome": 0},
        {"name": "Resolves YES (vol)", "x0": 0.55, "Q": 5e-4, "R": 5e-3, "outcome": 1},
        {"name": "Resolves NO (vol)", "x0": 0.45, "Q": 5e-4, "R": 5e-3, "outcome": 0},
        {"name": "Resolves YES (quiet)", "x0": 0.80, "Q": 1e-6, "R": 5e-4, "outcome": 1},
        {"name": "Resolves NO (noisy)", "x0": 0.50, "Q": 1e-4, "R": 1e-2, "outcome": 0},
    ]

    all_results = []
    all_preds = []
    all_outcomes = []

    for i, sc in enumerate(scenarios):
        data = generate_random_walk(n_steps=300, Q=sc["Q"], R=sc["R"], x0=sc["x0"], seed=42+i)
        bt = FilterBacktest(Q=sc["Q"], R=sc["R"])
        results = bt.run(data.observations, outcome=sc["outcome"])
        for r in results:
            all_results.append({"Filter": r.filter_name, "Brier": r.brier, "LogLoss": r.logloss})

        kf = ScalarKalmanFilter(Q=sc["Q"], R=sc["R"])
        kf_result = kf.filter(data.observations)
        all_preds.extend(kf_result.states)
        all_outcomes.extend([sc["outcome"]] * len(kf_result.states))

    # Brier score bar chart
    import pandas as pd
    df = pd.DataFrame(all_results)
    avg_brier = df.groupby("Filter")["Brier"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#FF6347", "#00CED1", "#9370DB", "#FFD700", "#888888"]
    bars = ax.barh(avg_brier.index, avg_brier.values, color=colors[:len(avg_brier)])
    ax.set_xlabel("Average Brier Score (lower is better)")
    ax.set_title("Filter Comparison: Average Brier Score")
    for bar, val in zip(bars, avg_brier.values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    save(fig, "fig_brier_comparison.pdf")

    # Print results
    avg_ll = df.groupby("Filter")["LogLoss"].mean().sort_values()
    print("  Avg Brier scores:")
    for name, val in avg_brier.items():
        print(f"    {name}: {val:.4f}")

    # Calibration curve
    all_preds = np.array(all_preds)
    all_outcomes = np.array(all_outcomes)
    bins, actual, counts = calibration_curve(all_preds, all_outcomes, n_bins=10)

    fig2, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.plot(bins, actual, "o-", color="#00CED1", markersize=8, label="Kalman filter")
    for b, a, c in zip(bins, actual, counts):
        if c > 0:
            ax.annotate(f"n={int(c)}", (b, a), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, alpha=0.7)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    save(fig2, "fig_calibration.pdf")


# ── Run all ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures for paper...\n")
    fig_filtered_vs_raw()
    fig_kalman_gain()
    fig_parameter_sensitivity()
    fig_likelihood_surface()
    fig_adaptive_comparison()
    fig_logit_bounded()
    fig_asymmetric_ci()
    fig_correlation_heatmap()
    fig_regime_detection()
    fig_backtest()
    print(f"\nDone! All figures saved to {FIGURES_DIR}/")
