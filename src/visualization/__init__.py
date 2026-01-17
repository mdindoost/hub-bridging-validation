"""
Visualization Module
====================

This module provides plotting and table generation utilities
for validation experiment results.

Submodules
----------
plots
    Matplotlib-based plotting functions
tables
    Summary table generation
"""

from .plots import (
    plot_rho_vs_h,
    plot_property_comparison,
    plot_algorithm_performance,
    plot_sparsification_effect,
    plot_calibration_curve,
    plot_degree_distribution,
    plot_experiment_5_results,
    save_figure,
)
from .tables import (
    create_summary_table,
    create_experiment_table,
    results_to_latex,
    results_to_markdown,
)

__all__ = [
    # Plots
    "plot_rho_vs_h",
    "plot_property_comparison",
    "plot_algorithm_performance",
    "plot_sparsification_effect",
    "plot_calibration_curve",
    "plot_degree_distribution",
    "plot_experiment_5_results",
    "save_figure",
    # Tables
    "create_summary_table",
    "create_experiment_table",
    "results_to_latex",
    "results_to_markdown",
]
