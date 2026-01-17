"""
Plotting Module
===============

This module provides matplotlib-based plotting functions for
visualizing validation experiment results.

All functions return matplotlib Figure objects for flexibility.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

logger = logging.getLogger(__name__)

# Set default style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")


def plot_rho_vs_h(
    h_values: Union[List[float], NDArray[np.float64]],
    rho_samples: NDArray[np.float64],
    fit_coeffs: Optional[List[float]] = None,
    title: str = "Hub-Bridging Ratio vs Parameter h",
    figsize: Tuple[int, int] = (8, 6),
    show_individual: bool = False,
) -> plt.Figure:
    """
    Plot hub-bridging ratio (rho_HB) as a function of parameter h.

    Parameters
    ----------
    h_values : array-like
        Values of h parameter
    rho_samples : NDArray
        rho_HB samples (n_h, n_samples)
    fit_coeffs : List[float], optional
        Polynomial fit coefficients for overlay
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size
    show_individual : bool, optional
        If True, show individual sample points

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    h_values = np.array(h_values)
    rho_mean = np.nanmean(rho_samples, axis=1)
    rho_std = np.nanstd(rho_samples, axis=1)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean with error bars
    ax.errorbar(
        h_values, rho_mean, yerr=rho_std,
        fmt="o-", capsize=5, capthick=2, linewidth=2, markersize=8,
        label="Mean ± Std Dev"
    )

    # Show individual points if requested
    if show_individual:
        for i, h in enumerate(h_values):
            samples = rho_samples[i][~np.isnan(rho_samples[i])]
            ax.scatter(
                np.full_like(samples, h), samples,
                alpha=0.3, s=20, color="gray"
            )

    # Overlay polynomial fit
    if fit_coeffs is not None:
        h_fine = np.linspace(h_values.min(), h_values.max(), 100)
        rho_fit = np.polyval(fit_coeffs, h_fine)
        ax.plot(h_fine, rho_fit, "--", linewidth=2, color="red",
                label=f"Polynomial fit (degree {len(fit_coeffs) - 1})")

    ax.set_xlabel("Hub-bridging parameter h", fontsize=12)
    ax.set_ylabel("Hub-bridging ratio ρ_HB", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_property_comparison(
    real_properties: Dict[str, Dict],
    synthetic_properties: Dict[float, Dict],
    property_name: str = "hub_bridging_ratio",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot comparison of a property between real and synthetic networks.

    Parameters
    ----------
    real_properties : Dict[str, Dict]
        Properties of real networks (name -> properties)
    synthetic_properties : Dict[float, Dict]
        Properties of synthetic networks (h -> properties)
    property_name : str, optional
        Property to compare
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract property values
    h_values = sorted(synthetic_properties.keys())

    # Get synthetic values
    synth_values = []
    for h in h_values:
        props = synthetic_properties[h]
        if property_name == "hub_bridging_ratio":
            val = props.get("hub_bridging", {}).get("rho_hb", np.nan)
        elif property_name == "modularity":
            val = props.get("community", {}).get("modularity", np.nan)
        elif property_name == "clustering":
            val = props.get("clustering", {}).get("global", np.nan)
        else:
            val = np.nan
        synth_values.append(val)

    # Plot synthetic trend
    ax.plot(h_values, synth_values, "o-", linewidth=2, markersize=8,
            label="Synthetic (HB-LFR)")

    # Plot real network values as horizontal lines
    colors = plt.cm.tab10(np.linspace(0, 1, len(real_properties)))
    for i, (name, props) in enumerate(real_properties.items()):
        if property_name == "hub_bridging_ratio":
            val = props.get("hub_bridging", {}).get("rho_hb", np.nan)
        elif property_name == "modularity":
            val = props.get("community", {}).get("modularity", np.nan)
        elif property_name == "clustering":
            val = props.get("clustering", {}).get("global", np.nan)
        else:
            val = np.nan

        if not np.isnan(val):
            ax.axhline(y=val, linestyle="--", color=colors[i], alpha=0.7,
                      label=f"{name}: {val:.3f}")

    ax.set_xlabel("Hub-bridging parameter h", fontsize=12)
    ax.set_ylabel(_get_property_label(property_name), fontsize=12)
    ax.set_title(title or f"{property_name.replace('_', ' ').title()} Comparison",
                 fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _get_property_label(property_name: str) -> str:
    """Get human-readable label for property."""
    labels = {
        "hub_bridging_ratio": "Hub-bridging ratio ρ_HB",
        "modularity": "Modularity Q",
        "clustering": "Clustering coefficient",
        "path_length": "Average path length",
    }
    return labels.get(property_name, property_name.replace("_", " ").title())


def plot_algorithm_performance(
    results: Dict[str, Any],
    metric: str = "nmi",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot community detection algorithm performance vs h.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from experiment_7_community_detection
    metric : str, optional
        Metric to plot: 'nmi' or 'ari' (default: 'nmi')
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    h_values = results["h_values"]
    algorithms = results["algorithms"]
    summary = results["performance_summary"]

    markers = ["o", "s", "^", "D", "v"]

    for i, alg in enumerate(algorithms):
        means = [summary[alg][h][f"{metric}_mean"] for h in h_values]
        stds = [summary[alg][h][f"{metric}_std"] for h in h_values]

        ax.errorbar(
            h_values, means, yerr=stds,
            fmt=f"{markers[i % len(markers)]}-",
            capsize=4, linewidth=2, markersize=7,
            label=alg.capitalize()
        )

    ax.set_xlabel("Hub-bridging parameter h", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title or f"Community Detection Performance ({metric.upper()})",
                 fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return fig


def plot_sparsification_effect(
    results: Dict[str, Any],
    method: str = "dspar",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Plot sparsification effects on community detection.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from experiment_8_sparsification
    method : str, optional
        Sparsification method to plot
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    h_values = results["h_values"]
    ratios = results["ratios"]
    summary = results["summary"][method]

    # Plot 1: NMI vs sparsification ratio for different h
    ax1 = axes[0]
    for h in h_values:
        nmis = [summary[r][h]["community_nmi_mean"] for r in ratios]
        ax1.plot(ratios, nmis, "o-", linewidth=2, markersize=6, label=f"h={h}")

    ax1.set_xlabel("Edge retention ratio", fontsize=12)
    ax1.set_ylabel("NMI (community recovery)", fontsize=12)
    ax1.set_title(f"{method.upper()}: Community Preservation", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Inter vs Intra retention
    ax2 = axes[1]
    for h in h_values:
        inter = [summary[r][h]["inter_retention_mean"] for r in ratios]
        intra = [summary[r][h]["intra_retention_mean"] for r in ratios]

        ax2.plot(ratios, inter, "o-", linewidth=2, label=f"Inter (h={h})")
        ax2.plot(ratios, intra, "s--", linewidth=2, alpha=0.7, label=f"Intra (h={h})")

    ax2.set_xlabel("Edge retention ratio", fontsize=12)
    ax2.set_ylabel("Edge type retention", fontsize=12)
    ax2.set_title(f"{method.upper()}: Edge Type Retention", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title or f"Sparsification Analysis ({method.upper()})", fontsize=14)
    fig.tight_layout()
    return fig


def plot_calibration_curve(
    calibration_result: Dict[str, Any],
    title: str = "Calibration Curve: h vs ρ_HB",
    figsize: Tuple[int, int] = (8, 6),
) -> plt.Figure:
    """
    Plot calibration curve from calibration results.

    Parameters
    ----------
    calibration_result : Dict[str, Any]
        Result from calibrate_h_to_rho
    title : str, optional
        Plot title
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    h_values = calibration_result["h_values"]
    rho_mean = calibration_result["rho_mean"]
    rho_std = calibration_result["rho_std"]

    # Plot data points with error bars
    ax.errorbar(
        h_values, rho_mean, yerr=rho_std,
        fmt="o", capsize=5, capthick=2, markersize=8,
        label="Measured"
    )

    # Plot interpolation curve if available
    interpolator = calibration_result.get("interpolator")
    if interpolator is not None:
        h_fine = np.linspace(min(h_values), max(h_values), 100)
        rho_fine = interpolator(h_fine)
        ax.plot(h_fine, rho_fine, "-", linewidth=2, alpha=0.7,
                label="Interpolation")

    ax.set_xlabel("Hub-bridging parameter h", fontsize=12)
    ax.set_ylabel("Hub-bridging ratio ρ_HB", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_degree_distribution(
    degrees: NDArray[np.int64],
    title: str = "Degree Distribution",
    log_scale: bool = True,
    fit_powerlaw: bool = True,
    figsize: Tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot degree distribution with optional power-law fit.

    Parameters
    ----------
    degrees : NDArray[np.int64]
        Array of node degrees
    title : str, optional
        Plot title
    log_scale : bool, optional
        If True, use log-log scale
    fit_powerlaw : bool, optional
        If True, overlay power-law fit
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Histogram
    ax1 = axes[0]
    ax1.hist(degrees, bins=50, density=True, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Degree", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Degree Histogram", fontsize=12)

    # Right: CCDF (complementary cumulative distribution)
    ax2 = axes[1]

    # Compute CCDF
    sorted_degrees = np.sort(degrees)[::-1]
    ccdf = np.arange(1, len(sorted_degrees) + 1) / len(sorted_degrees)

    ax2.scatter(sorted_degrees, ccdf, alpha=0.5, s=10)

    if log_scale:
        ax2.set_xscale("log")
        ax2.set_yscale("log")

    # Fit power-law if requested
    if fit_powerlaw:
        try:
            import powerlaw
            fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin

            # Plot fit line
            x_fit = np.logspace(np.log10(xmin), np.log10(max(degrees)), 100)
            y_fit = (x_fit / xmin) ** (-alpha + 1)
            y_fit = y_fit * ccdf[np.searchsorted(-sorted_degrees, -xmin)]

            ax2.plot(x_fit, y_fit, "r--", linewidth=2,
                    label=f"Power-law (α={alpha:.2f})")
            ax2.legend()
        except Exception as e:
            logger.debug(f"Power-law fit failed: {e}")

    ax2.set_xlabel("Degree", fontsize=12)
    ax2.set_ylabel("P(X ≥ x)", fontsize=12)
    ax2.set_title("Complementary CDF", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def create_multi_panel_figure(
    plots: List[Tuple[str, plt.Figure]],
    n_cols: int = 2,
    figsize_per_plot: Tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Combine multiple figures into a multi-panel figure.

    Parameters
    ----------
    plots : List[Tuple[str, plt.Figure]]
        List of (title, figure) tuples
    n_cols : int, optional
        Number of columns
    figsize_per_plot : Tuple[int, int], optional
        Size per subplot

    Returns
    -------
    plt.Figure
        Combined figure
    """
    n_plots = len(plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (title, plot_fig) in enumerate(plots):
        # This is a simplified approach - in practice would need to
        # properly copy axes content
        axes[i].set_title(title)
        axes[i].text(0.5, 0.5, f"Plot: {title}", ha="center", va="center")

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    filepath: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
) -> None:
    """
    Save figure to file.

    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filepath : str
        Output path (extension determines format)
    dpi : int, optional
        Resolution (default: 300)
    bbox_inches : str, optional
        Bounding box (default: 'tight')
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Saved figure to {filepath}")
