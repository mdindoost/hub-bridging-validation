"""
Tables Module
=============

This module provides functions for generating summary tables
from validation experiment results in various formats.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_summary_table(
    results: Dict[str, Any],
    experiment_type: str,
) -> pd.DataFrame:
    """
    Create a summary table from experiment results.

    Parameters
    ----------
    results : Dict[str, Any]
        Experiment results dictionary
    experiment_type : str
        Type of experiment: 'parameter_control', 'degree_preservation',
        'community_detection', 'sparsification'

    Returns
    -------
    pd.DataFrame
        Summary table as DataFrame
    """
    if experiment_type == "parameter_control":
        return _create_parameter_control_table(results)
    elif experiment_type == "degree_preservation":
        return _create_degree_preservation_table(results)
    elif experiment_type == "community_detection":
        return _create_community_detection_table(results)
    elif experiment_type == "sparsification":
        return _create_sparsification_table(results)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")


def _create_parameter_control_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create table for Experiment 1 results."""
    h_values = results["h_values"]
    rho_mean = results["rho_mean"]
    rho_std = results["rho_std"]
    rho_ci_lower = results.get("rho_ci_lower", rho_mean - 1.96 * rho_std)
    rho_ci_upper = results.get("rho_ci_upper", rho_mean + 1.96 * rho_std)

    df = pd.DataFrame({
        "h": h_values,
        "rho_HB_mean": rho_mean,
        "rho_HB_std": rho_std,
        "95%_CI_lower": rho_ci_lower,
        "95%_CI_upper": rho_ci_upper,
    })

    # Add monotonicity test results
    mono_test = results.get("monotonicity_test", {})
    df.attrs["is_monotonic"] = mono_test.get("is_monotonic", "N/A")
    df.attrs["spearman_r"] = mono_test.get("spearman_r", "N/A")

    return df


def _create_degree_preservation_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create table for Experiment 2 results."""
    h_values = results["h_values"]
    degree_stats = results["degree_stats"]
    ks_tests = results.get("ks_tests", {})

    rows = []
    for h in h_values:
        stats = degree_stats.get(h, {})
        ks = ks_tests.get(h, {})

        rows.append({
            "h": h,
            "mean_degree": stats.get("mean_degree", np.nan),
            "std_degree": stats.get("std_degree", np.nan),
            "max_degree": stats.get("max_degree", np.nan),
            "KS_statistic": ks.get("statistic", np.nan),
            "KS_p_value": ks.get("p_value", np.nan),
        })

    return pd.DataFrame(rows)


def _create_community_detection_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create table for Experiment 7 results."""
    h_values = results["h_values"]
    algorithms = results["algorithms"]
    summary = results["performance_summary"]

    rows = []
    for alg in algorithms:
        for h in h_values:
            perf = summary[alg][h]
            rows.append({
                "algorithm": alg,
                "h": h,
                "NMI_mean": perf["nmi_mean"],
                "NMI_std": perf["nmi_std"],
                "ARI_mean": perf["ari_mean"],
                "ARI_std": perf["ari_std"],
                "n_samples": perf["n_samples"],
            })

    return pd.DataFrame(rows)


def _create_sparsification_table(results: Dict[str, Any]) -> pd.DataFrame:
    """Create table for Experiment 8 results."""
    h_values = results["h_values"]
    methods = results["methods"]
    ratios = results["ratios"]
    summary = results["summary"]

    rows = []
    for method in methods:
        for ratio in ratios:
            for h in h_values:
                data = summary[method][ratio][h]
                rows.append({
                    "method": method,
                    "retention_ratio": ratio,
                    "h": h,
                    "community_NMI": data["community_nmi_mean"],
                    "inter_retention": data["inter_retention_mean"],
                    "intra_retention": data["intra_retention_mean"],
                })

    return pd.DataFrame(rows)


def create_experiment_table(
    experiments: Dict[str, Dict[str, Any]],
    metrics: List[str],
) -> pd.DataFrame:
    """
    Create a combined table from multiple experiments.

    Parameters
    ----------
    experiments : Dict[str, Dict[str, Any]]
        Dictionary of experiment_name -> results
    metrics : List[str]
        Metrics to include in table

    Returns
    -------
    pd.DataFrame
        Combined table
    """
    rows = []

    for exp_name, results in experiments.items():
        row = {"experiment": exp_name}

        for metric in metrics:
            if metric in results:
                value = results[metric]
                if isinstance(value, (int, float)):
                    row[metric] = value
                elif isinstance(value, dict):
                    row[metric] = str(value)
                else:
                    row[metric] = str(value)
            else:
                row[metric] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def results_to_latex(
    df: pd.DataFrame,
    caption: str = "Results Table",
    label: str = "tab:results",
    float_format: str = "%.3f",
) -> str:
    """
    Convert DataFrame to LaTeX table format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    caption : str, optional
        Table caption
    label : str, optional
        LaTeX label
    float_format : str, optional
        Format string for floats

    Returns
    -------
    str
        LaTeX table code
    """
    latex = df.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=True,
    )

    # Clean up column names (replace underscores with spaces)
    for col in df.columns:
        latex = latex.replace(col, col.replace("_", " "))

    return latex


def results_to_markdown(
    df: pd.DataFrame,
    float_format: str = "%.3f",
) -> str:
    """
    Convert DataFrame to Markdown table format.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    float_format : str, optional
        Format string for floats

    Returns
    -------
    str
        Markdown table
    """
    # Format floats
    df_formatted = df.copy()
    for col in df_formatted.select_dtypes(include=[np.floating]).columns:
        df_formatted[col] = df_formatted[col].apply(lambda x: float_format % x if not np.isnan(x) else "N/A")

    return df_formatted.to_markdown(index=False)


def create_property_comparison_table(
    real_properties: Dict[str, Dict],
    synthetic_properties: Dict[float, Dict],
    properties_to_compare: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Create comparison table between real and synthetic network properties.

    Parameters
    ----------
    real_properties : Dict[str, Dict]
        Properties of real networks
    synthetic_properties : Dict[float, Dict]
        Properties of synthetic networks at different h values
    properties_to_compare : List[str], optional
        Which properties to include

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    if properties_to_compare is None:
        properties_to_compare = [
            "rho_hb", "modularity", "clustering", "mean_degree"
        ]

    rows = []

    # Add real networks
    for name, props in real_properties.items():
        row = {"network": name, "type": "real"}
        for prop in properties_to_compare:
            row[prop] = _extract_property(props, prop)
        rows.append(row)

    # Add synthetic networks
    for h, props in sorted(synthetic_properties.items()):
        row = {"network": f"HB-LFR (h={h})", "type": "synthetic"}
        for prop in properties_to_compare:
            row[prop] = _extract_property(props, prop)
        rows.append(row)

    return pd.DataFrame(rows)


def _extract_property(props: Dict, prop_name: str) -> float:
    """Extract a property value from nested dictionary."""
    if prop_name == "rho_hb":
        return props.get("hub_bridging", {}).get("rho_hb", np.nan)
    elif prop_name == "modularity":
        return props.get("community", {}).get("modularity", np.nan)
    elif prop_name == "clustering":
        return props.get("clustering", {}).get("global", np.nan)
    elif prop_name == "mean_degree":
        return props.get("degree", {}).get("mean", np.nan)
    elif prop_name == "path_length":
        return props.get("path_length", {}).get("average_path_length", np.nan)
    else:
        return np.nan


def create_validation_summary(
    experiment_results: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Create overall validation summary table.

    Parameters
    ----------
    experiment_results : Dict[str, Dict[str, Any]]
        Results from all experiments

    Returns
    -------
    pd.DataFrame
        Summary table with pass/fail status for each validation
    """
    rows = []

    # Experiment 1: Parameter Control
    if "experiment_1" in experiment_results:
        res = experiment_results["experiment_1"]
        rows.append({
            "Experiment": "1. Parameter Control",
            "Description": "h controls ρ_HB monotonically",
            "Status": "PASS" if res.get("monotonicity_test", {}).get("is_monotonic") else "FAIL",
            "Key Metric": f"Spearman r = {res.get('spearman_correlation', {}).get('r', np.nan):.3f}",
        })

    # Experiment 2: Degree Preservation
    if "experiment_2" in experiment_results:
        res = experiment_results["experiment_2"]
        rows.append({
            "Experiment": "2. Degree Preservation",
            "Description": "Degree distribution preserved across h",
            "Status": "PASS" if res.get("preservation_passed") else "FAIL",
            "Key Metric": f"KS tests passed",
        })

    # Experiment 3: Modularity Independence
    if "experiment_3" in experiment_results:
        res = experiment_results["experiment_3"]
        rows.append({
            "Experiment": "3. Modularity Independence",
            "Description": "Q independent of h (controlled by μ)",
            "Status": "PASS" if res.get("all_independent") else "FAIL",
            "Key Metric": "Kruskal-Wallis tests",
        })

    # Experiment 4: Concentration
    if "experiment_4" in experiment_results:
        res = experiment_results["experiment_4"]
        rows.append({
            "Experiment": "4. Concentration",
            "Description": "ρ_HB has low variance",
            "Status": "PASS" if res.get("all_concentrated") else "FAIL",
            "Key Metric": f"CV < {res.get('cv_threshold', 0.1)}",
        })

    return pd.DataFrame(rows)


def save_table(
    df: pd.DataFrame,
    filepath: str,
    format: str = "csv",
) -> None:
    """
    Save table to file.

    Parameters
    ----------
    df : pd.DataFrame
        Table to save
    filepath : str
        Output path
    format : str, optional
        Output format: 'csv', 'latex', 'markdown', 'excel'
    """
    if format == "csv":
        df.to_csv(filepath, index=False)
    elif format == "latex":
        with open(filepath, "w") as f:
            f.write(results_to_latex(df))
    elif format == "markdown":
        with open(filepath, "w") as f:
            f.write(results_to_markdown(df))
    elif format == "excel":
        df.to_excel(filepath, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    logger.info(f"Saved table to {filepath}")
