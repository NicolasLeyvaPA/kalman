"""Visualization functions for Kalman filter analysis.

Provides standardized plots for filtered vs raw prices, Kalman gain evolution,
innovation diagnostics, parameter sensitivity analysis, and SNR improvement.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from src.filters.scalar_kalman import KalmanResult, ScalarKalmanFilter

# Consistent plot style
PLOT_STYLE: dict[str, Any] = {
    "figure.figsize": (12, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

# Color palette
COLOR_RAW: str = "#888888"       # Gray for raw observations
COLOR_FILTERED: str = "#00CED1"  # Cyan/teal for filtered estimate
COLOR_CONFIDENCE: str = "#00CED1"
CONFIDENCE_ALPHA: float = 0.15
CONFIDENCE_SIGMA: float = 2.0    # Number of standard deviations for bands


def plot_filtered_vs_raw(
    result: KalmanResult,
    title: str = "Kalman Filtered Probability",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot filtered estimate vs raw observations with confidence bands.

    Parameters
    ----------
    result : KalmanResult
        Output from running the Kalman filter.
    title : str
        Plot title.
    ax : plt.Axes or None
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    Figure or None
        The matplotlib Figure if a new one was created, else None.
    """
    with plt.rc_context(PLOT_STYLE):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        x_axis = np.arange(len(result.observations))
        if result.timestamps is not None:
            x_axis = result.timestamps

        # Raw observations
        ax.plot(x_axis, result.observations, color=COLOR_RAW, alpha=0.5,
                linewidth=0.8, label="Raw price")

        # Filtered estimate
        ax.plot(x_axis, result.states, color=COLOR_FILTERED, linewidth=1.5,
                label="Kalman estimate")

        # Confidence band
        std = np.sqrt(result.covariances) * CONFIDENCE_SIGMA
        ax.fill_between(
            x_axis,
            result.states - std,
            result.states + std,
            color=COLOR_CONFIDENCE,
            alpha=CONFIDENCE_ALPHA,
            label=f"\u00b1{CONFIDENCE_SIGMA:.0f}\u03c3 confidence",
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.set_ylim(-0.05, 1.05)

        if fig is not None:
            fig.tight_layout()
        return fig


def plot_kalman_gain(
    result: KalmanResult,
    title: str = "Kalman Gain Over Time",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot the Kalman gain evolution showing convergence to steady state.

    Parameters
    ----------
    result : KalmanResult
        Output from running the Kalman filter.
    title : str
        Plot title.
    ax : plt.Axes or None
        Axes to plot on.

    Returns
    -------
    Figure or None
        The matplotlib Figure if created.
    """
    with plt.rc_context(PLOT_STYLE):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        x_axis = np.arange(len(result.gains))
        ax.plot(x_axis, result.gains, color=COLOR_FILTERED, linewidth=1.5)

        # Annotate steady state
        if len(result.gains) > 10:
            ss_gain = result.gains[-1]
            ax.axhline(y=ss_gain, color="red", linestyle="--", alpha=0.5)
            ax.annotate(
                f"Gain \u2192 {ss_gain:.4f}",
                xy=(len(result.gains) - 1, ss_gain),
                xytext=(-100, 20),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "red"},
                color="red",
            )

        ax.set_xlabel("Time step")
        ax.set_ylabel("Kalman Gain")
        ax.set_title(title)

        if fig is not None:
            fig.tight_layout()
        return fig


def plot_innovations(
    result: KalmanResult,
    title: str = "Innovation Sequence (Filter Diagnostics)",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plot the innovation sequence with confidence bounds.

    A well-tuned filter should have ~95% of innovations within the
    +/-2 sigma bounds derived from the innovation covariance S_t.

    Parameters
    ----------
    result : KalmanResult
        Output from running the Kalman filter.
    title : str
        Plot title.
    ax : plt.Axes or None
        Axes to plot on.

    Returns
    -------
    Figure or None
        The matplotlib Figure if created.
    """
    with plt.rc_context(PLOT_STYLE):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        x_axis = np.arange(len(result.innovations))
        ax.plot(x_axis, result.innovations, color=COLOR_FILTERED, alpha=0.7,
                linewidth=0.8, label="Innovation")

        # Bounds from innovation covariance
        if result.innovation_covariances is not None:
            bounds = CONFIDENCE_SIGMA * np.sqrt(result.innovation_covariances)
            ax.fill_between(x_axis, -bounds, bounds, color="red",
                            alpha=0.1, label=f"\u00b1{CONFIDENCE_SIGMA:.0f}\u03c3 bounds")

            # Compute fraction within bounds
            within = np.abs(result.innovations) <= bounds
            frac_within = within.mean()
            ax.set_title(f"{title}\n({frac_within:.1%} within bounds, expect ~95%)")
        else:
            ax.set_title(title)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Innovation")
        ax.legend(loc="best")

        if fig is not None:
            fig.tight_layout()
        return fig


def plot_parameter_sensitivity(
    observations: np.ndarray,
    Q_values: list[float] | None = None,
    R_values: list[float] | None = None,
    fixed_Q: float = 1e-4,
    fixed_R: float = 1e-3,
) -> Figure:
    """Show how filtered output changes with different Q and R values.

    Creates a 2-panel figure: left varies Q with fixed R, right varies R with fixed Q.

    Parameters
    ----------
    observations : np.ndarray
        Raw observation sequence.
    Q_values : list[float] or None
        Q values to test. Default: [1e-6, 1e-5, 1e-4, 1e-3].
    R_values : list[float] or None
        R values to test. Default: [1e-4, 1e-3, 1e-2, 1e-1].
    fixed_Q : float
        Q value when varying R.
    fixed_R : float
        R value when varying Q.

    Returns
    -------
    Figure
        Two-panel sensitivity plot.
    """
    if Q_values is None:
        Q_values = [1e-6, 1e-5, 1e-4, 1e-3]
    if R_values is None:
        R_values = [1e-4, 1e-3, 1e-2, 1e-1]

    with plt.rc_context(PLOT_STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = sns.color_palette("viridis", max(len(Q_values), len(R_values)))

        # Left: vary Q
        ax1.plot(observations, color=COLOR_RAW, alpha=0.3, linewidth=0.5, label="Raw")
        for i, Q in enumerate(Q_values):
            kf = ScalarKalmanFilter(Q=Q, R=fixed_R)
            result = kf.filter(observations)
            ax1.plot(result.states, color=colors[i], linewidth=1.2,
                     label=f"Q={Q:.0e}")
        ax1.set_title(f"Varying Q (R={fixed_R:.0e})")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Probability")
        ax1.legend(loc="best", fontsize=9)

        # Right: vary R
        ax2.plot(observations, color=COLOR_RAW, alpha=0.3, linewidth=0.5, label="Raw")
        for i, R in enumerate(R_values):
            kf = ScalarKalmanFilter(Q=fixed_Q, R=R)
            result = kf.filter(observations)
            ax2.plot(result.states, color=colors[i], linewidth=1.2,
                     label=f"R={R:.0e}")
        ax2.set_title(f"Varying R (Q={fixed_Q:.0e})")
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Probability")
        ax2.legend(loc="best", fontsize=9)

        fig.tight_layout()
        return fig


def plot_snr_improvement(
    result: KalmanResult,
    title: str = "Signal-to-Noise Ratio Improvement",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Show variance reduction achieved by the filter.

    Computes the variance of raw price changes vs filtered estimate changes
    and displays the improvement ratio.

    Parameters
    ----------
    result : KalmanResult
        Output from running the Kalman filter.
    title : str
        Plot title.
    ax : plt.Axes or None
        Axes to plot on.

    Returns
    -------
    Figure or None
        The matplotlib Figure if created.
    """
    with plt.rc_context(PLOT_STYLE):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        raw_var = np.var(np.diff(result.observations))
        filtered_var = np.var(np.diff(result.states))
        snr_ratio = raw_var / (filtered_var + 1e-15)

        labels = ["Raw Price\nVariance", "Filtered\nVariance"]
        values = [raw_var, filtered_var]
        bar_colors = [COLOR_RAW, COLOR_FILTERED]

        bars = ax.bar(labels, values, color=bar_colors, width=0.5)
        ax.set_ylabel("Variance of First Differences")
        ax.set_title(f"{title}\nSNR improvement: {snr_ratio:.1f}x")

        # Annotate bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2e}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        if fig is not None:
            fig.tight_layout()
        return fig


def plot_likelihood_surface(
    Q_range: np.ndarray,
    R_range: np.ndarray,
    log_likelihood_grid: np.ndarray,
    Q_hat: float | None = None,
    R_hat: float | None = None,
    title: str = "Log-Likelihood Surface",
) -> Figure:
    """Plot the log-likelihood as a heatmap over Q and R parameter space.

    Parameters
    ----------
    Q_range : np.ndarray
        Array of Q values tested.
    R_range : np.ndarray
        Array of R values tested.
    log_likelihood_grid : np.ndarray
        2D array of log-likelihood values (shape: len(Q_range) x len(R_range)).
    Q_hat : float or None
        MLE estimate of Q (marked on plot if provided).
    R_hat : float or None
        MLE estimate of R (marked on plot if provided).
    title : str
        Plot title.

    Returns
    -------
    Figure
        Heatmap of the log-likelihood surface.
    """
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.pcolormesh(
            np.log10(R_range), np.log10(Q_range), log_likelihood_grid,
            shading="auto", cmap="viridis",
        )
        fig.colorbar(im, ax=ax, label="Log-Likelihood")

        if Q_hat is not None and R_hat is not None:
            ax.plot(np.log10(R_hat), np.log10(Q_hat), "r*", markersize=15,
                    label=f"MLE: Q={Q_hat:.2e}, R={R_hat:.2e}")
            ax.legend(loc="best")

        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(Q)")
        ax.set_title(title)
        fig.tight_layout()
        return fig
