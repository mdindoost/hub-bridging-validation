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


def plot_degree_preservation_comparison(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    Visualize degree distribution preservation results from Experiment 2.

    Creates 3-panel figure:
    - Panel A: τ estimates vs h (both generators)
    - Panel B: Degree distribution examples (h=0 vs h=2)
    - Panel C: Mean degree stability

    Parameters
    ----------
    results : Dict[str, Any]
        Results from experiment_2_degree_preservation_full()
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Determine available generators
    generators = [g for g in ['hb_sbm', 'hb_lfr'] if g in results]
    if not generators:
        logger.warning("No generator results found")
        return fig

    h_values = results[generators[0]]['h_values']
    colors = {'hb_sbm': '#2E86AB', 'hb_lfr': '#A23B72'}
    markers = {'hb_sbm': 'o', 'hb_lfr': 's'}

    # Panel A: τ estimates vs h
    ax = axes[0]

    for gen_name in generators:
        gen_res = results[gen_name]
        stats = gen_res.get('statistics', {})
        tau_means = stats.get('tau_means', {})
        tau_stds = stats.get('tau_stds', {})

        means = [tau_means.get(h, np.nan) for h in h_values]
        stds = [tau_stds.get(h, np.nan) for h in h_values]

        # Filter out NaN values
        valid_h = [h for i, h in enumerate(h_values) if not np.isnan(means[i])]
        valid_means = [m for m in means if not np.isnan(m)]
        valid_stds = [s for s in stds if not np.isnan(s)]

        if valid_means:
            ax.errorbar(valid_h, valid_means, yerr=valid_stds,
                       fmt=f'{markers[gen_name]}-', capsize=5, capthick=2,
                       label=gen_name.upper(),
                       color=colors[gen_name], linewidth=2, markersize=8)

    # Target line
    target = stats.get('target_tau1', 2.5)
    ax.axhline(y=target, color='gray', linestyle='--',
              alpha=0.5, label=f'Target τ={target}')

    # Acceptable range
    ax.axhspan(target - 0.3, target + 0.3, alpha=0.1, color='green',
              label='Acceptable range (±0.3)')

    ax.set_xlabel('Hub-bridging parameter h', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power-law exponent τ', fontsize=12, fontweight='bold')
    ax.set_title('Panel A: Degree Exponent Preservation',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([1.5, 4.0])

    # Panel B: Example degree distributions (log-log)
    ax = axes[1]

    # Plot degree distributions for h=0 and h=max
    h_compare = [h_values[0], h_values[-1]]  # First and last h values
    line_styles = {h_values[0]: '-', h_values[-1]: '--'}
    alphas = {h_values[0]: 0.7, h_values[-1]: 1.0}

    for gen_name in generators:
        gen_res = results[gen_name]
        degree_seqs = gen_res.get('degree_sequences', {})

        for h in h_compare:
            if h in degree_seqs and degree_seqs[h]:
                # Use first sample's degree sequence
                degrees = degree_seqs[h][0]

                # Compute degree distribution
                unique, counts = np.unique(degrees, return_counts=True)
                prob = counts / counts.sum()

                label = f"{gen_name.upper()} (h={h:.1f})"
                ax.loglog(unique, prob, marker=markers[gen_name],
                         linestyle=line_styles[h],
                         label=label, color=colors[gen_name],
                         alpha=alphas[h], markersize=5, linewidth=1.5)

    # Reference power-law line
    x_ref = np.logspace(0.3, 2, 50)
    y_ref = x_ref**(-2.5)
    y_ref = y_ref / y_ref.max() * 0.3  # Scale for visibility
    ax.loglog(x_ref, y_ref, 'k:', alpha=0.4, linewidth=2, label='τ=2.5 reference')

    ax.set_xlabel('Degree k', fontsize=12, fontweight='bold')
    ax.set_ylabel('P(k)', fontsize=12, fontweight='bold')
    ax.set_title(f'Panel B: Degree Distributions\n(h={h_values[0]} vs h={h_values[-1]})',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.3)

    # Panel C: Mean degree stability
    ax = axes[2]

    for gen_name in generators:
        gen_res = results[gen_name]
        mean_degs = gen_res.get('mean_degrees', {})

        means = []
        stds = []
        valid_h = []

        for h in h_values:
            if h in mean_degs and mean_degs[h]:
                valid_h.append(h)
                means.append(np.mean(mean_degs[h]))
                stds.append(np.std(mean_degs[h]))

        if means:
            ax.errorbar(valid_h, means, yerr=stds,
                       fmt=f'{markers[gen_name]}-', capsize=5, capthick=2,
                       label=gen_name.upper(),
                       color=colors[gen_name], linewidth=2, markersize=8)

    ax.set_xlabel('Hub-bridging parameter h', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean degree ⟨k⟩', fontsize=12, fontweight='bold')
    ax.set_title('Panel C: Mean Degree Stability',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle('Experiment 2: Degree Distribution Preservation',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to: {save_path}")

        # Also save PDF
        pdf_path = save_path.replace('.png', '.pdf')
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"PDF saved to: {pdf_path}")

    return fig


def plot_degree_preservation_summary_table(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create a summary table visualization for Experiment 2 results.

    Parameters
    ----------
    results : Dict[str, Any]
        Results from experiment_2_degree_preservation_full()
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object with table
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    generators = [g for g in ['hb_sbm', 'hb_lfr'] if g in results]
    h_values = results[generators[0]]['h_values']

    # Build table data
    headers = ['h'] + [f'{g.upper()} τ' for g in generators]
    table_data = [headers]

    for h in h_values:
        row = [f'{h:.2f}']
        for gen_name in generators:
            stats = results[gen_name].get('statistics', {})
            tau_means = stats.get('tau_means', {})
            tau_stds = stats.get('tau_stds', {})
            mean = tau_means.get(h, np.nan)
            std = tau_stds.get(h, np.nan)
            if not np.isnan(mean):
                row.append(f'{mean:.2f} ± {std:.2f}')
            else:
                row.append('N/A')
        table_data.append(row)

    # Add summary row
    summary_row = ['Overall']
    for gen_name in generators:
        stats = results[gen_name].get('statistics', {})
        overall_mean = stats.get('tau_overall_mean', np.nan)
        overall_std = stats.get('tau_overall_std', np.nan)
        if not np.isnan(overall_mean):
            summary_row.append(f'{overall_mean:.2f} ± {overall_std:.2f}')
        else:
            summary_row.append('N/A')
    table_data.append(summary_row)

    # Add validation status
    status_row = ['Status']
    for gen_name in generators:
        stats = results[gen_name].get('statistics', {})
        passes = stats.get('passes', False)
        status_row.append('✓ PASS' if passes else '✗ FAIL')
    table_data.append(status_row)

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center',
                    loc='center', colWidths=[0.15] + [0.25] * len(generators))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header row
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')

    # Style summary row
    summary_row_idx = len(h_values) + 1
    for i in range(len(headers)):
        table[(summary_row_idx, i)].set_facecolor('#F5F5F5')
        table[(summary_row_idx, i)].set_text_props(weight='bold')

    # Style status row
    status_row_idx = len(h_values) + 2
    for i, gen_name in enumerate(generators, 1):
        stats = results[gen_name].get('statistics', {})
        passes = stats.get('passes', False)
        color = '#C8E6C9' if passes else '#FFCDD2'
        table[(status_row_idx, i)].set_facecolor(color)
        table[(status_row_idx, i)].set_text_props(weight='bold')

    ax.set_title('Experiment 2: Degree Exponent (τ) Summary\n',
                fontsize=14, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Table saved to: {save_path}")

    return fig


def plot_experiment_5_results(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 10),
) -> plt.Figure:
    """
    Visualize Experiment 5: Real Network Property Matching results.

    Creates multi-panel figure:
    - Panel A: Property distance comparison (Real, HB-LFR, Standard LFR)
    - Panel B: Fitted h values per network
    - Panel C: ρ_HB comparison (Real vs Fitted HB-LFR)
    - Panel D: Key properties comparison radar chart

    Parameters
    ----------
    results : Dict[str, Any]
        Results from experiment_5_real_network_matching()
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Results are keyed by network name directly (not under "networks")
    # Filter out non-network keys like "summary", "metadata"
    network_names = [k for k in results.keys() if k not in ("summary", "metadata") and isinstance(results[k], dict)]
    if not network_names:
        logger.warning("No network results found")
        return fig

    n_networks = len(network_names)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_networks, 3)))

    # Panel A: Property Distance Comparison (Bar chart)
    ax = axes[0, 0]

    x = np.arange(n_networks)
    width = 0.25

    hb_distances = []
    lfr_distances = []
    improvements = []

    for name in network_names:
        net_res = results[name]
        hb_dist = net_res.get("overall_distance_hb", np.nan)
        lfr_dist = net_res.get("overall_distance_std", np.nan)
        hb_distances.append(hb_dist)
        lfr_distances.append(lfr_dist)
        if not np.isnan(hb_dist) and not np.isnan(lfr_dist) and lfr_dist > 0:
            improvements.append((lfr_dist - hb_dist) / lfr_dist * 100)
        else:
            imp = net_res.get("overall_improvement", 0)
            improvements.append(imp * 100 if not np.isnan(imp) else 0)

    bars1 = ax.bar(x - width/2, hb_distances, width, label='HB-LFR (fitted)',
                   color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, lfr_distances, width, label='Standard LFR',
                   color='#A23B72', alpha=0.8)

    ax.set_xlabel('Network', fontsize=12, fontweight='bold')
    ax.set_ylabel('Property Distance to Real', fontsize=12, fontweight='bold')
    ax.set_title('Panel A: Property Matching Quality\n(Lower is Better)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(network_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add improvement percentage annotations
    for i, (hb, lfr, imp) in enumerate(zip(hb_distances, lfr_distances, improvements)):
        if imp > 0:
            ax.annotate(f'+{imp:.0f}%', xy=(i, max(hb, lfr)),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=9, color='green', fontweight='bold')

    # Panel B: Fitted h values
    ax = axes[0, 1]

    fitted_h = []
    target_rho = []
    achieved_rho = []

    for name in network_names:
        net_res = results[name]
        fitted_h.append(net_res.get("h_fitted", np.nan))
        # Get target rho from fit_result or real_properties
        fit_result = net_res.get("fit_result", {})
        real_props = net_res.get("real_properties", {})
        target_rho.append(fit_result.get("rho_target", real_props.get("rho_HB", np.nan)))
        achieved_rho.append(fit_result.get("rho_achieved", np.nan))

    bars = ax.bar(network_names, fitted_h, color=colors[:n_networks], alpha=0.8)

    ax.set_xlabel('Network', fontsize=12, fontweight='bold')
    ax.set_ylabel('Fitted h Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Panel B: Optimal h Values\n(Higher h → More Hub-Bridging)',
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(network_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, h in zip(bars, fitted_h):
        if not np.isnan(h):
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')

    # Panel C: ρ_HB Comparison
    ax = axes[1, 0]

    x = np.arange(n_networks)
    width = 0.35

    bars1 = ax.bar(x - width/2, target_rho, width, label='Real Network',
                   color='#27AE60', alpha=0.8)
    bars2 = ax.bar(x + width/2, achieved_rho, width, label='HB-LFR (fitted)',
                   color='#2E86AB', alpha=0.8)

    ax.set_xlabel('Network', fontsize=12, fontweight='bold')
    ax.set_ylabel('Hub-Bridging Ratio ρ_HB', fontsize=12, fontweight='bold')
    ax.set_title('Panel C: ρ_HB Target vs Achieved\n(Should Match)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(network_names, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add error annotations
    for i, (t, a) in enumerate(zip(target_rho, achieved_rho)):
        if not np.isnan(t) and not np.isnan(a):
            error = abs(a - t)
            ax.annotate(f'Δ={error:.3f}', xy=(i, max(t, a)),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8, color='gray')

    # Panel D: Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary = results.get("summary", {})

    # Create summary text
    summary_text = "EXPERIMENT 5 SUMMARY\n"
    summary_text += "=" * 40 + "\n\n"

    summary_text += f"Networks tested: {summary.get('n_networks', 'N/A')}\n\n"

    # Statistical test results
    stat_test = summary.get("statistical_test", {})
    if stat_test:
        summary_text += "Mann-Whitney U Test (HB-LFR vs Standard LFR):\n"
        summary_text += f"  U-statistic: {stat_test.get('U_statistic', np.nan):.2f}\n"
        summary_text += f"  p-value: {stat_test.get('p_value', np.nan):.4f}\n"
        summary_text += f"  Effect size: {stat_test.get('effect_size', np.nan):.3f}\n\n"

    # Average improvement
    avg_improvement = summary.get("avg_improvement_percent", np.nan)
    if not np.isnan(avg_improvement):
        summary_text += f"Average improvement: {avg_improvement:.1f}%\n"
        summary_text += f"Networks where HB-LFR wins: {summary.get('hb_wins', 0)}/{summary.get('n_networks', 0)}\n\n"

    # Validation status
    passes = summary.get("passes", False)
    status = "✓ PASS" if passes else "✗ FAIL"
    color = "green" if passes else "red"
    summary_text += f"\nVALIDATION: {status}\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add interpretation
    interpretation = (
        "Interpretation:\n"
        "• HB-LFR with fitted h should produce smaller property\n"
        "  distances than standard LFR (h=0)\n"
        "• Significant p-value (<0.05) indicates HB-LFR\n"
        "  systematically outperforms standard LFR\n"
        "• This validates that real networks exhibit\n"
        "  hub-bridging structure that h can capture"
    )
    ax.text(0.1, 0.25, interpretation, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', style='italic',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.suptitle('Experiment 5: Real Network Property Matching\n'
                'Validating HB-LFR Realism with Fitted h Parameter',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to: {save_path}")

        # Also save PDF
        pdf_path = save_path.replace('.png', '.pdf')
        if pdf_path != save_path:
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"PDF saved to: {pdf_path}")

    return fig


def plot_modularity_independence(
    results: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    Visualize modularity independence results for Experiment 4.

    Creates 3-panel figure:
    - Panel A: Q vs ρ_HB scatter plot with correlation
    - Panel B: Q vs h with error bars (should be flat)
    - Panel C: Q coefficient of variation across h values

    Parameters
    ----------
    results : Dict[str, Any]
        Results from experiment_4_modularity_independence()
    save_path : str, optional
        Path to save figure
    figsize : Tuple[int, int], optional
        Figure size

    Returns
    -------
    plt.Figure
        Matplotlib figure object
    """
    from scipy.stats import linregress

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    colors = {'hb_sbm': '#2E86AB', 'hb_lfr': '#A23B72'}

    for gen_name, gen_res in results.items():
        color = colors.get(gen_name, '#666666')
        h_values = gen_res['h_values']

        # Panel A: Scatter plot Q vs ρ_HB
        ax = axes[0]

        all_rho = [rho for h in h_values for rho in gen_res['rho_samples'][h]]
        all_Q = [Q for h in h_values for Q in gen_res['Q_samples'][h]]

        ax.scatter(all_rho, all_Q, alpha=0.4, s=30,
                  label=gen_name.upper(), color=color)

        # Add regression line
        slope, intercept, r_value, p_value, std_err = linregress(all_rho, all_Q)
        x_line = np.array([min(all_rho), max(all_rho)])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, '--', color=color, linewidth=2, alpha=0.6,
               label=f'{gen_name.upper()} fit (r={r_value:+.3f})')

        # Panel B: Q vs h (should be flat)
        ax_b = axes[1]

        Q_means = [gen_res['statistics']['Q_means'][h] for h in h_values]
        Q_stds = [gen_res['statistics']['Q_stds'][h] for h in h_values]

        ax_b.errorbar(h_values, Q_means, yerr=Q_stds,
                     fmt='o-', capsize=5, capthick=2, markersize=8,
                     label=gen_name.upper(), color=color, linewidth=2)

        # Add horizontal line at mean Q
        mean_Q = np.mean(Q_means)
        ax_b.axhline(mean_Q, color=color, linestyle='--', alpha=0.5)

        # Panel C: Coefficient of variation
        ax_c = axes[2]

        cv_values = [gen_res['statistics']['Q_cv_by_h'][h] * 100 for h in h_values]
        mean_cv = gen_res['statistics']['mean_cv'] * 100

        x_pos = np.arange(len(h_values))
        ax_c.bar(x_pos, cv_values, color=color, alpha=0.7, label=gen_name.upper())
        ax_c.axhline(mean_cv, color=color, linestyle='--', linewidth=2,
                    label=f'Mean CV={mean_cv:.1f}%')

    # Format Panel A
    axes[0].set_xlabel('Hub-bridging ratio ρ_HB', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Modularity Q', fontsize=12, fontweight='bold')
    axes[0].set_title('Panel A: Q vs ρ_HB\n(Testing Independence)',
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Add text box with correlation for first generator
    gen_name = list(results.keys())[0]
    textstr = f'Pearson r = {results[gen_name]["statistics"]["pearson_r"]:+.3f}\n'
    textstr += f'p = {results[gen_name]["statistics"]["pearson_p"]:.4f}\n'
    if abs(results[gen_name]["statistics"]["pearson_r"]) < 0.2:
        textstr += 'Independent ✓'
    elif abs(results[gen_name]["statistics"]["pearson_r"]) < 0.3:
        textstr += 'Weak dependence'
    else:
        textstr += 'Dependent ⚠'

    props = dict(boxstyle='round', facecolor=colors.get(gen_name, '#666666'), alpha=0.2)
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes,
                fontsize=10, verticalalignment='top', bbox=props)

    # Format Panel B
    axes[1].set_xlabel('Hub-bridging parameter h', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Modularity Q', fontsize=12, fontweight='bold')
    axes[1].set_title('Panel B: Q vs h\n(Should be Flat)',
                     fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Format Panel C
    axes[2].set_xlabel('h value', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Panel C: Q Variability\nacross h values',
                     fontsize=14, fontweight='bold')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels([f'{h:.1f}' for h in h_values])
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Experiment 4: Modularity Independence Test\n'
                'Validating Theorem 4(a): Q independent of ρ_HB',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Figure saved to: {save_path}")

        # Also save PDF
        pdf_path = save_path.replace('.png', '.pdf')
        if pdf_path != save_path:
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"PDF saved to: {pdf_path}")

    return fig
