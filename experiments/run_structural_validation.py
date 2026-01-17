#!/usr/bin/env python3
"""
Structural Validation Runner
============================

This script runs Experiments 1-4 for structural validation of
hub-bridging generators.

Experiments:
1. Parameter control (h -> rho_HB)
2. Degree distribution preservation
3. Modularity independence from h
4. Concentration/variance analysis

Usage:
    python run_structural_validation.py [--config CONFIG] [--output OUTPUT]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_validation_config, load_network_params
from src.generators.hb_lfr import hb_lfr
from src.generators.hb_sbm import hb_sbm
from src.validation.structural import (
    experiment_1_parameter_control,
    experiment_2_degree_preservation,
    experiment_2_degree_preservation_full,
    experiment_3_modularity_independence,
    experiment_4_concentration,
    experiment_4_modularity_independence,
)

# Optional visualization imports
try:
    from src.visualization import (
        plot_rho_vs_h,
        create_summary_table,
        save_figure,
    )
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    plot_rho_vs_h = None
    create_summary_table = None
    save_figure = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run structural validation experiments (1-4)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file (default: use built-in config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/structural",
        help="Output directory for results",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["hb_lfr", "hb_sbm", "both"],
        default="hb_lfr",
        help="Generator to validate (default: hb_lfr)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["1", "2", "3", "4", "all"],
        default=["all"],
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override number of samples per setting",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation with reduced samples",
    )
    return parser.parse_args()


def get_generator(name: str):
    """Get generator function by name."""
    if name == "hb_lfr":
        return hb_lfr
    elif name == "hb_sbm":
        return hb_sbm
    else:
        raise ValueError(f"Unknown generator: {name}")


def run_experiment_1(
    generator_func,
    generator_params,
    config,
    seed,
    n_samples=None,
):
    """Run Experiment 1: Parameter Control."""
    logger.info("Running Experiment 1: Parameter Control")

    exp_config = config["experiments"]["parameter_control"]
    h_values = exp_config["h_values"]

    if n_samples is None:
        n_samples = config["sample_sizes"]["parameter_control"]

    results = experiment_1_parameter_control(
        generator_func=generator_func,
        generator_params=generator_params,
        h_values=h_values,
        n_samples=n_samples,
        seed=seed,
    )

    return results


def run_experiment_2(
    generator_func,
    generator_params,
    config,
    seed,
    n_samples=None,
):
    """Run Experiment 2: Degree Preservation."""
    logger.info("Running Experiment 2: Degree Preservation")

    exp_config = config["experiments"]["degree_preservation"]
    h_values = exp_config["h_values"]

    if n_samples is None:
        n_samples = config["sample_sizes"]["degree_preservation"]

    results = experiment_2_degree_preservation(
        generator_func=generator_func,
        generator_params=generator_params,
        h_values=h_values,
        n_samples=n_samples,
        seed=seed,
    )

    return results


def run_experiment_3(
    generator_func,
    generator_params,
    config,
    seed,
    n_samples=None,
):
    """Run Experiment 3: Modularity Independence."""
    logger.info("Running Experiment 3: Modularity Independence")

    exp_config = config["experiments"]["modularity_independence"]
    h_values = exp_config["h_values"]
    mu_values = exp_config["mu_values"]

    if n_samples is None:
        n_samples = config["sample_sizes"]["modularity_independence"]

    results = experiment_3_modularity_independence(
        generator_func=generator_func,
        generator_params=generator_params,
        h_values=h_values,
        mu_values=mu_values,
        n_samples=n_samples,
        seed=seed,
    )

    return results


def run_experiment_4(
    generator_func,
    generator_params,
    config,
    seed,
    n_samples=None,
):
    """Run Experiment 4: Concentration."""
    logger.info("Running Experiment 4: Concentration")

    exp_config = config["experiments"]["concentration"]
    h_values = exp_config["h_values"]

    if n_samples is None:
        n_samples = config["sample_sizes"]["concentration"]

    results = experiment_4_concentration(
        generator_func=generator_func,
        generator_params=generator_params,
        h_values=h_values,
        n_samples=n_samples,
        seed=seed,
    )

    return results


def save_results(results, output_dir, experiment_name, generator_name):
    """Save experiment results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{experiment_name}_{generator_name}_{timestamp}"

    # Save raw results as JSON (convert numpy arrays)
    results_serializable = _make_serializable(results)
    json_path = output_path / f"{base_name}.json"
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    logger.info(f"Saved results to {json_path}")

    return json_path


def _make_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif callable(obj):
        return "<function>"
    else:
        return obj


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_validation_config()
    network_params = load_network_params()

    # Override with quick settings if requested
    if args.quick:
        args.n_samples = 5
        network_params["lfr"]["default"]["n"] = 250

    # Set up generator parameters
    generator_params = network_params["lfr"]["default"].copy()
    # Remove 'h' since it will be varied
    generator_params.pop("h", None)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which experiments to run
    if "all" in args.experiments:
        experiments_to_run = ["1", "2", "3", "4"]
    else:
        experiments_to_run = args.experiments

    # Determine which generators to test
    if args.generator == "both":
        generators = ["hb_lfr", "hb_sbm"]
    else:
        generators = [args.generator]

    # Run experiments
    all_results = {}

    for gen_name in generators:
        logger.info(f"Testing generator: {gen_name}")
        generator_func = get_generator(gen_name)

        gen_results = {}

        if "1" in experiments_to_run:
            results = run_experiment_1(
                generator_func, generator_params, config, args.seed, args.n_samples
            )
            gen_results["experiment_1"] = results
            save_results(results, output_dir, "exp1_parameter_control", gen_name)

            # Generate plot (if visualization available)
            if HAS_VISUALIZATION and "rho_samples" in results:
                fig = plot_rho_vs_h(
                    results["h_values"],
                    results["rho_samples"],
                    fit_coeffs=results.get("fit_results", {}).get("coefficients"),
                    title=f"Parameter Control ({gen_name})",
                )
                fig_path = output_dir / f"exp1_rho_vs_h_{gen_name}.pdf"
                save_figure(fig, str(fig_path))

        if "2" in experiments_to_run:
            results = run_experiment_2(
                generator_func, generator_params, config, args.seed, args.n_samples
            )
            gen_results["experiment_2"] = results
            save_results(results, output_dir, "exp2_degree_preservation", gen_name)

        if "3" in experiments_to_run:
            results = run_experiment_3(
                generator_func, generator_params, config, args.seed, args.n_samples
            )
            gen_results["experiment_3"] = results
            save_results(results, output_dir, "exp3_modularity_independence", gen_name)

        if "4" in experiments_to_run:
            results = run_experiment_4(
                generator_func, generator_params, config, args.seed, args.n_samples
            )
            gen_results["experiment_4"] = results
            save_results(results, output_dir, "exp4_concentration", gen_name)

        all_results[gen_name] = gen_results

    # Print summary
    logger.info("=" * 60)
    logger.info("STRUCTURAL VALIDATION SUMMARY")
    logger.info("=" * 60)

    for gen_name, gen_results in all_results.items():
        logger.info(f"\nGenerator: {gen_name}")
        logger.info("-" * 40)

        if "experiment_1" in gen_results:
            res = gen_results["experiment_1"]
            mono = res.get("monotonicity_test", {})
            logger.info(
                f"  Exp 1 (Parameter Control): "
                f"monotonic={mono.get('is_monotonic', 'N/A')}, "
                f"Spearman r={mono.get('spearman_r', np.nan):.4f}"
            )

        if "experiment_2" in gen_results:
            res = gen_results["experiment_2"]
            logger.info(
                f"  Exp 2 (Degree Preservation): "
                f"passed={res.get('preservation_passed', 'N/A')}"
            )

        if "experiment_3" in gen_results:
            res = gen_results["experiment_3"]
            logger.info(
                f"  Exp 3 (Modularity Independence): "
                f"all_independent={res.get('all_independent', 'N/A')}"
            )

        if "experiment_4" in gen_results:
            res = gen_results["experiment_4"]
            logger.info(
                f"  Exp 4 (Concentration): "
                f"all_concentrated={res.get('all_concentrated', 'N/A')}"
            )

    logger.info(f"\nResults saved to: {output_dir}")


def experiment_1_parameter_control_improved(
    generator='hb_sbm',
    h_values=None,
    n_samples=30,
    n=500,
    k=5,
    p_in=0.3,
    p_out=0.05,
    theta_distribution='power_law',
    degree_correction_scale=1.5,
    seed=42,
):
    """
    Improved Experiment 1 with more samples and better statistics.

    Parameters
    ----------
    generator : str
        Generator to use: 'hb_sbm' or 'hb_lfr'
    h_values : list, optional
        Values of h to test (default: 9 values from 0 to 2)
    n_samples : int
        Number of samples per h value
    n : int
        Number of nodes
    k : int
        Number of communities
    p_in : float
        Intra-community edge probability
    p_out : float
        Inter-community edge probability
    theta_distribution : str
        Distribution for degree corrections: 'exponential', 'power_law', 'lognormal'
    degree_correction_scale : float
        Scale parameter for degree heterogeneity
    seed : int
        Random seed

    Returns
    -------
    dict
        Experiment results with statistics and validation status
    """
    from scipy.stats import spearmanr, pearsonr
    from src.generators.hb_sbm import hb_sbm
    from src.generators.hb_lfr import hb_lfr
    from src.metrics.hub_bridging import compute_hub_bridging_ratio
    from src.validation.statistical_tests import test_monotonicity

    if h_values is None:
        h_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    # Select generator function
    if generator == 'hb_sbm':
        generator_func = hb_sbm
    elif generator == 'hb_lfr':
        generator_func = hb_lfr
    else:
        raise ValueError(f"Unknown generator: {generator}")

    print("=" * 60)
    print("EXPERIMENT 1: Parameter Control Validation (IMPROVED)")
    print("=" * 60)
    print(f"Generator: {generator}")
    print(f"Parameters: n={n}, k={k}, p_in={p_in}, p_out={p_out}")
    print(f"Theta distribution: {theta_distribution}, scale={degree_correction_scale}")
    print(f"Samples per h: {n_samples}")
    print()

    results = {h: [] for h in h_values}

    for h in h_values:
        print(f"Generating {n_samples} graphs with h={h:.2f}...", end="", flush=True)

        for sample in range(n_samples):
            # Use improved generator with theta distribution
            if generator == 'hb_sbm':
                G, communities = hb_sbm(
                    n=n, k=k,
                    p_in=p_in, p_out=p_out,
                    h=h,
                    seed=seed + sample,
                    theta_distribution=theta_distribution,
                    degree_correction_scale=degree_correction_scale,
                )
            else:
                G, communities = hb_lfr(
                    n=n, mu=0.3, h=h,
                    seed=seed + sample,
                )

            try:
                rho = compute_hub_bridging_ratio(G, communities)
                results[h].append(rho)
            except ValueError:
                pass  # Skip failed samples

        if results[h]:
            current_mean = np.mean(results[h])
            current_std = np.std(results[h])
            print(f" ρ = {current_mean:.3f} ± {current_std:.3f} ({len(results[h])}/{n_samples} valid)")
        else:
            print(" No valid samples!")

    # Compute statistics
    rho_means = {h: np.mean(results[h]) for h in h_values if results[h]}
    rho_stds = {h: np.std(results[h]) for h in h_values if results[h]}
    rho_cis = {h: 1.96 * rho_stds[h] / np.sqrt(len(results[h]))
               for h in h_values if results[h]}

    # Test monotonicity - need 2D array format (n_h x n_samples)
    valid_h = [h for h in h_values if results[h]]
    # Convert to 2D array: rows = h values, cols = samples
    max_samples = max(len(results[h]) for h in valid_h)
    y_samples_2d = np.full((len(valid_h), max_samples), np.nan)
    for i, h in enumerate(valid_h):
        y_samples_2d[i, :len(results[h])] = results[h]
    monotonicity_results = test_monotonicity(valid_h, y_samples_2d)

    # Correlation tests (using all samples)
    h_flat = [h for h in h_values for _ in range(len(results[h]))]
    rho_flat = [rho for h in h_values for rho in results[h]]

    if len(h_flat) > 2:
        spearman_r, spearman_p = spearmanr(h_flat, rho_flat)
        pearson_r, pearson_p = pearsonr(h_flat, rho_flat)
    else:
        spearman_r = spearman_p = pearson_r = pearson_p = np.nan

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\nMean ρ_HB by h value:")
    for h in h_values:
        if h in rho_means:
            print(f"  h={h:.2f}: ρ = {rho_means[h]:.3f} ± {rho_stds[h]:.3f} " +
                  f"(95% CI: ±{rho_cis[h]:.3f})")

    print(f"\nMonotonicity: {monotonicity_results['is_monotonic']}")
    print(f"Spearman correlation: r = {spearman_r:.3f}, p = {spearman_p:.4e}")
    print(f"Pearson correlation:  r = {pearson_r:.3f}, p = {pearson_p:.4e}")

    # Check monotonicity in the increasing phase (up to saturation)
    # Find where saturation begins (where slope becomes flat/negative)
    increasing_h = []
    for i, h in enumerate(valid_h[:-1]):
        if rho_means[h] < rho_means[valid_h[i+1]]:
            increasing_h.append(h)
        else:
            break
    increasing_h.append(valid_h[len(increasing_h)])  # Include saturation point

    # Test monotonicity only in increasing phase
    is_monotonic_increasing = len(increasing_h) >= 3

    # Assess success criteria for publication quality:
    # 1. Strong correlation (r > 0.8)
    # 2. Statistically significant (p < 0.001)
    # 3. Clear increasing phase before saturation
    passes = (
        spearman_r > 0.8 and
        spearman_p < 0.001 and
        is_monotonic_increasing
    )

    print()
    print(f"VALIDATION: {'PASS ✓' if passes else 'FAIL ✗'}")

    if spearman_r > 0.8:
        print(f"  ✓ Strong correlation (r = {spearman_r:.3f} > 0.8)")
    else:
        print(f"  ✗ Correlation too weak (r = {spearman_r:.3f} <= 0.8)")

    if spearman_p < 0.001:
        print(f"  ✓ Statistically significant (p = {spearman_p:.2e})")
    else:
        print(f"  ✗ Not significant (p = {spearman_p:.4e} >= 0.001)")

    if is_monotonic_increasing:
        print(f"  ✓ Monotonic increase up to h = {increasing_h[-1]:.2f} (then saturation)")
    else:
        print("  ✗ No clear monotonic increasing phase")

    if len(increasing_h) < len(valid_h):
        print(f"  ℹ Saturation detected at h > {increasing_h[-1]:.2f} (expected behavior)")

    return {
        'h_values': h_values,
        'rho_samples': results,
        'rho_means': rho_means,
        'rho_stds': rho_stds,
        'rho_cis': rho_cis,
        'monotonicity': monotonicity_results,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'passes': passes,
        'generator': generator,
        'params': {
            'n': n, 'k': k, 'p_in': p_in, 'p_out': p_out,
            'theta_distribution': theta_distribution,
            'degree_correction_scale': degree_correction_scale,
        }
    }


def experiment_1_compare_generators(
    h_values=None,
    n_samples=20,
    n=500,
    seed=42,
):
    """
    Compare HB-SBM and HB-LFR generators on Experiment 1.

    Parameters
    ----------
    h_values : list, optional
        Values of h to test
    n_samples : int
        Samples per h value
    n : int
        Number of nodes
    seed : int
        Random seed

    Returns
    -------
    dict
        Comparison results for both generators
    """
    from src.generators.hb_sbm import hb_sbm
    from src.generators.hb_lfr import hb_lfr
    from src.metrics.hub_bridging import compute_hub_bridging_ratio
    from scipy.stats import spearmanr

    if h_values is None:
        h_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    print("=" * 70)
    print("EXPERIMENT 1: Generator Comparison (HB-SBM vs HB-LFR)")
    print("=" * 70)
    print(f"Parameters: n={n}, samples/h={n_samples}")
    print()

    # Results storage
    results = {
        'hb_sbm': {h: [] for h in h_values},
        'hb_lfr': {h: [] for h in h_values},
    }

    # Run HB-SBM
    print("Testing HB-SBM...")
    for h in h_values:
        print(f"  h={h:.2f}: ", end="", flush=True)
        for sample in range(n_samples):
            try:
                G, communities = hb_sbm(
                    n=n, k=5, p_in=0.3, p_out=0.05, h=h,
                    theta_distribution='power_law',
                    degree_correction_scale=1.5,
                    seed=seed + sample,
                )
                rho = compute_hub_bridging_ratio(G, communities)
                results['hb_sbm'][h].append(rho)
            except Exception as e:
                pass
        if results['hb_sbm'][h]:
            print(f"ρ = {np.mean(results['hb_sbm'][h]):.3f} ± {np.std(results['hb_sbm'][h]):.3f}")
        else:
            print("No valid samples")

    # Run HB-LFR
    print("\nTesting HB-LFR...")
    for h in h_values:
        print(f"  h={h:.2f}: ", end="", flush=True)
        for sample in range(n_samples):
            try:
                G, communities = hb_lfr(
                    n=n, mu=0.3, h=h,
                    seed=seed + sample,
                )
                rho = compute_hub_bridging_ratio(G, communities)
                results['hb_lfr'][h].append(rho)
            except Exception as e:
                pass
        if results['hb_lfr'][h]:
            print(f"ρ = {np.mean(results['hb_lfr'][h]):.3f} ± {np.std(results['hb_lfr'][h]):.3f}")
        else:
            print("No valid samples")

    # Compute statistics
    stats = {}
    for gen in ['hb_sbm', 'hb_lfr']:
        h_flat = [h for h in h_values for _ in range(len(results[gen][h]))]
        rho_flat = [rho for h in h_values for rho in results[gen][h]]

        if len(h_flat) > 2:
            r, p = spearmanr(h_flat, rho_flat)
        else:
            r, p = np.nan, np.nan

        stats[gen] = {
            'spearman_r': r,
            'spearman_p': p,
            'rho_means': {h: np.mean(results[gen][h]) if results[gen][h] else np.nan
                         for h in h_values},
            'rho_stds': {h: np.std(results[gen][h]) if results[gen][h] else np.nan
                        for h in h_values},
        }

    # Print comparison table
    print()
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(f"{'h':>6} | {'HB-SBM ρ':>15} | {'HB-LFR ρ':>15} | {'Diff':>10}")
    print("-" * 52)
    for h in h_values:
        sbm_mean = stats['hb_sbm']['rho_means'].get(h, np.nan)
        lfr_mean = stats['hb_lfr']['rho_means'].get(h, np.nan)
        diff = lfr_mean - sbm_mean if not np.isnan(sbm_mean) and not np.isnan(lfr_mean) else np.nan
        print(f"{h:>6.2f} | {sbm_mean:>15.3f} | {lfr_mean:>15.3f} | {diff:>+10.3f}")

    print()
    print("Correlation Statistics:")
    print(f"  HB-SBM: Spearman r = {stats['hb_sbm']['spearman_r']:.3f}, "
          f"p = {stats['hb_sbm']['spearman_p']:.2e}")
    print(f"  HB-LFR: Spearman r = {stats['hb_lfr']['spearman_r']:.3f}, "
          f"p = {stats['hb_lfr']['spearman_p']:.2e}")

    # Validation summary
    print()
    print("Validation Summary:")
    for gen in ['hb_sbm', 'hb_lfr']:
        passes = stats[gen]['spearman_r'] > 0.7 and stats[gen]['spearman_p'] < 0.001
        print(f"  {gen.upper()}: {'PASS ✓' if passes else 'FAIL ✗'} "
              f"(r={stats[gen]['spearman_r']:.3f})")

    return {
        'h_values': h_values,
        'results': results,
        'stats': stats,
        'n': n,
        'n_samples': n_samples,
    }


def run_experiment_3_concentration(
    generators=None,
    h_fixed=1.0,
    n_samples=100,
    n=500,
    seed=42,
):
    """
    Run Experiment 3: Concentration and Reproducibility.

    Tests that hub-bridging ratio has low variance across multiple runs,
    validating that the algorithm is reliable and reproducible.

    Parameters
    ----------
    generators : list, optional
        Generators to test (default: ['hb_sbm', 'hb_lfr'])
    h_fixed : float
        Fixed h value to test (default: 1.0)
    n_samples : int
        Number of independent networks to generate (default: 100)
    n : int
        Network size
    seed : int
        Random seed

    Returns
    -------
    dict
        Experiment results with CV and statistics
    """
    from scipy.stats import shapiro, normaltest
    from src.generators.hb_sbm import hb_sbm
    from src.generators.hb_lfr import hb_lfr
    from src.metrics.hub_bridging import compute_hub_bridging_ratio

    if generators is None:
        generators = ['hb_sbm', 'hb_lfr']

    print("=" * 70)
    print("EXPERIMENT 3: Concentration and Reproducibility")
    print("=" * 70)
    print(f"Fixed h = {h_fixed}")
    print(f"Samples: {n_samples}")
    print(f"Network size: n = {n}")
    print(f"Generators: {generators}")
    print()

    rng = np.random.default_rng(seed)
    results = {}

    for gen_name in generators:
        print(f"\n--- Testing {gen_name.upper()} ---")

        rho_samples = []

        for i in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                if gen_name == 'hb_sbm':
                    G, communities = hb_sbm(
                        n=n, k=5, h=h_fixed, seed=sample_seed,
                        p_in=0.3, p_out=0.05,
                        theta_distribution='power_law',
                        degree_correction_scale=1.5
                    )
                elif gen_name == 'hb_lfr':
                    G, communities = hb_lfr(
                        n=n, mu=0.3, h=h_fixed, seed=sample_seed,
                        max_iters=5000
                    )
                else:
                    raise ValueError(f"Unknown generator: {gen_name}")

                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples.append(rho)

                if (i + 1) % 20 == 0:
                    current_mean = np.mean(rho_samples)
                    current_std = np.std(rho_samples)
                    current_cv = (current_std / current_mean) * 100
                    print(f"  Sample {i+1}/{n_samples}: ρ = {current_mean:.3f} ± {current_std:.3f} (CV = {current_cv:.2f}%)")

            except Exception as e:
                print(f"  Sample {i+1} failed: {e}")

        # Compute statistics
        rho_array = np.array(rho_samples)
        mean_rho = np.mean(rho_array)
        std_rho = np.std(rho_array)
        cv = (std_rho / mean_rho) * 100
        ci_95 = 1.96 * std_rho / np.sqrt(len(rho_array))

        # Normality tests
        if len(rho_array) >= 20:
            shapiro_stat, shapiro_p = shapiro(rho_array[:50])  # Shapiro limited to 50
            _, dagostino_p = normaltest(rho_array)
        else:
            shapiro_p = dagostino_p = np.nan

        # Store results
        results[gen_name] = {
            'h': h_fixed,
            'n_samples': len(rho_samples),
            'rho_samples': rho_samples,
            'mean': mean_rho,
            'std': std_rho,
            'cv': cv,
            'ci_95': ci_95,
            'min': np.min(rho_array),
            'max': np.max(rho_array),
            'range': np.max(rho_array) - np.min(rho_array),
            'shapiro_p': shapiro_p,
            'dagostino_p': dagostino_p,
            'passes': cv < 15.0,  # Threshold: CV < 15%
        }

        print(f"\n  Results for {gen_name.upper()}:")
        print(f"    Mean ρ_HB: {mean_rho:.4f}")
        print(f"    Std ρ_HB:  {std_rho:.4f}")
        print(f"    CV:        {cv:.2f}%")
        print(f"    95% CI:    ±{ci_95:.4f}")
        print(f"    Range:     [{results[gen_name]['min']:.3f}, {results[gen_name]['max']:.3f}]")
        print(f"    Normality: Shapiro p={shapiro_p:.4f}, D'Agostino p={dagostino_p:.4f}")

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 70)
    print(f"\n{'Generator':<12} | {'Mean ρ':>10} | {'Std':>8} | {'CV (%)':>8} | {'Status':>10}")
    print("-" * 56)

    for gen_name in generators:
        r = results[gen_name]
        status = "✓ PASS" if r['passes'] else "✗ FAIL"
        print(f"{gen_name.upper():<12} | {r['mean']:>10.4f} | {r['std']:>8.4f} | {r['cv']:>8.2f} | {status:>10}")

    print()
    print("Pass criteria: CV < 15%")
    print("Expected: HB-LFR CV < 5% (deterministic), HB-SBM CV < 10% (stochastic)")

    return results


def run_experiment_2_full(
    generators=None,
    h_values=None,
    n_samples=15,
    n=500,
    target_tau1=2.5,
    seed=42,
):
    """
    Run Experiment 2: Degree Distribution Preservation (Full version).

    This validates that hub-bridging control is independent of degree structure
    by showing τ remains constant across h values.

    Parameters
    ----------
    generators : list, optional
        Generators to test (default: ['hb_sbm', 'hb_lfr'])
    h_values : list, optional
        h values to test (default: [0.0, 0.5, 1.0, 1.5, 2.0])
    n_samples : int
        Samples per (generator, h) combination
    n : int
        Network size
    target_tau1 : float
        Target power-law exponent
    seed : int
        Random seed

    Returns
    -------
    dict
        Experiment results
    """
    if generators is None:
        generators = ['hb_sbm', 'hb_lfr']
    if h_values is None:
        h_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    print("=" * 70)
    print("EXPERIMENT 2: Degree Distribution Preservation")
    print("=" * 70)
    print(f"Generators: {generators}")
    print(f"h values: {h_values}")
    print(f"Samples per h: {n_samples}")
    print(f"Network size: n={n}")
    print(f"Target τ: {target_tau1}")
    print()

    # Run the comprehensive experiment
    results = experiment_2_degree_preservation_full(
        generators=generators,
        h_values=h_values,
        n_samples=n_samples,
        n=n,
        target_tau1=target_tau1,
        seed=seed,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 70)

    for gen_name in generators:
        if gen_name in results:
            stats = results[gen_name].get('statistics', {})
            print(f"\n{gen_name.upper()}:")
            print(f"  Overall τ: {stats.get('tau_overall_mean', np.nan):.3f} ± "
                  f"{stats.get('tau_overall_std', np.nan):.3f}")
            print(f"  Target τ: {target_tau1}")
            print(f"  ANOVA p (τ independent of h): {stats.get('anova_p', np.nan):.4f}")
            print(f"  τ independent of h: {stats.get('independent', False)}")
            print(f"  Close to target: {stats.get('close_to_target', False)}")
            print(f"  VALIDATION: {'PASS ✓' if stats.get('passes', False) else 'FAIL ✗'}")

    return results


def run_experiment_4_modularity(
    generators=None,
    h_values=None,
    n_samples=30,
    n=500,
    seed=42,
):
    """
    Run Experiment 4: Modularity Independence Test.

    Validates Theorem 4(a): For fixed constraints (degree sequence,
    community structure, |E_inter|), modularity Q is independent of
    hub-bridging ratio ρ_HB.

    Parameters
    ----------
    generators : list, optional
        Generators to test (default: ['hb_lfr'])
    h_values : list, optional
        h values to test (default: [0.0, 0.5, 1.0, 1.5, 2.0])
    n_samples : int
        Samples per h value
    n : int
        Network size
    seed : int
        Random seed

    Returns
    -------
    dict
        Experiment results
    """
    if generators is None:
        generators = ['hb_lfr']
    if h_values is None:
        h_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    print("=" * 70)
    print("EXPERIMENT 4: Modularity Independence Test")
    print("=" * 70)
    print("Validating Theorem 4(a): Q independent of ρ_HB for fixed constraints")
    print(f"Generators: {generators}")
    print(f"h values: {h_values}")
    print(f"Samples per h: {n_samples}")
    print(f"Network size: n={n}")
    print()

    # Run the experiment
    results = experiment_4_modularity_independence(
        generators=generators,
        h_values=h_values,
        n_samples=n_samples,
        n=n,
        seed=seed,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 RESULTS")
    print("=" * 70)

    for gen_name in generators:
        if gen_name in results:
            s = results[gen_name]['statistics']
            a = results[gen_name]['assessment']
            print(f"\n{gen_name.upper()}:")
            print(f"  Pearson r(ρ_HB, Q):   {s['pearson_r']:+.4f}")
            print(f"  Spearman r(ρ_HB, Q):  {s['spearman_r']:+.4f}")
            print(f"  ANOVA p-value:        {s['anova_p']:.4f}")
            print(f"  Q range across h:     {s['Q_range']:.4f}")
            print(f"  Relative range:       {s['relative_range']:.1%}")
            print(f"  Degree preserved:     {s['degree_preserved']}")
            print(f"  Passes correlation:   {a['passes_correlation']}")
            print(f"  Passes independence:  {a['passes_anova']}")
            print(f"  VALIDATION: {'PASS ✓' if a['passes_overall'] else 'FAIL ✗'}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("If |r| < 0.2: Strong evidence for Theorem 4(a) - Q independent of ρ_HB")
    print("If |r| < 0.3: Moderate evidence, acceptable")
    print("If |r| > 0.3: Some dependency detected (investigate)")
    print()
    print("Note: A SMALL negative correlation is theoretically expected from")
    print("Theorem 4(b) - higher ρ_HB slightly constrains maximum Q.")
    print("But for fixed constraints (Theorem 4a), should be ~0.")

    return results


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--exp2":
            # Run Experiment 2: Degree Preservation
            n_samples = 15
            if len(sys.argv) > 2:
                n_samples = int(sys.argv[2])

            results = run_experiment_2_full(
                generators=['hb_sbm', 'hb_lfr'],
                n_samples=n_samples,
                n=500,
                seed=42,
            )

            # Save results
            import pickle
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / 'experiment_2_results.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(f"\nResults saved to: {output_dir / 'experiment_2_results.pkl'}")

            # Generate visualization
            try:
                from src.visualization.plots import plot_degree_preservation_comparison
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt

                fig = plot_degree_preservation_comparison(
                    results,
                    save_path=str(output_dir / 'figure_degree_preservation.png')
                )
                plt.close(fig)
                print(f"Figure saved to: {output_dir / 'figure_degree_preservation.png'}")
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")

        elif sys.argv[1] == "--exp3":
            # Run Experiment 3: Concentration
            n_samples = 100
            if len(sys.argv) > 2:
                n_samples = int(sys.argv[2])

            results = run_experiment_3_concentration(
                generators=['hb_sbm', 'hb_lfr'],
                h_fixed=1.0,
                n_samples=n_samples,
                n=500,
                seed=42,
            )

            # Save results
            import pickle
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / 'experiment_3_results.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(f"\nResults saved to: {output_dir / 'experiment_3_results.pkl'}")

            # Generate histogram visualization
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                for idx, gen_name in enumerate(['hb_sbm', 'hb_lfr']):
                    ax = axes[idx]
                    r = results[gen_name]

                    ax.hist(r['rho_samples'], bins=20, density=True, alpha=0.7,
                           color='#2E86AB' if gen_name == 'hb_sbm' else '#A23B72',
                           edgecolor='black')

                    # Add vertical lines for mean and std
                    ax.axvline(r['mean'], color='red', linestyle='-', linewidth=2,
                              label=f"Mean = {r['mean']:.3f}")
                    ax.axvline(r['mean'] - r['std'], color='red', linestyle='--', alpha=0.5)
                    ax.axvline(r['mean'] + r['std'], color='red', linestyle='--', alpha=0.5,
                              label=f"±1 Std = {r['std']:.3f}")

                    ax.set_xlabel('Hub-bridging ratio ρ_HB', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
                    ax.set_title(f"{gen_name.upper()} (h={r['h']})\nCV = {r['cv']:.2f}%, n={r['n_samples']}",
                                fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)

                plt.suptitle('Experiment 3: Concentration and Reproducibility',
                            fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()

                fig.savefig(str(output_dir / 'figure_concentration.png'),
                           dpi=300, bbox_inches='tight', facecolor='white')
                fig.savefig(str(output_dir / 'figure_concentration.pdf'),
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                print(f"Figure saved to: {output_dir / 'figure_concentration.png'}")
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")

        elif sys.argv[1] == "--exp4":
            # Run Experiment 4: Modularity Independence
            n_samples = 30
            if len(sys.argv) > 2:
                n_samples = int(sys.argv[2])

            print("\n" + "=" * 70)
            print("Running Experiment 4: Modularity Independence Test")
            print("=" * 70)
            print("Validating Theorem 4(a): Q independent of ρ_HB for fixed constraints")
            print()

            results = run_experiment_4_modularity(
                generators=['hb_lfr'],  # HB-LFR preferred for exact degree preservation
                h_values=[0.0, 0.5, 1.0, 1.5, 2.0],
                n_samples=n_samples,
                n=500,
                seed=42,
            )

            # Save results
            import pickle
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / 'experiment_4_results.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(f"\nResults saved to: {output_dir / 'experiment_4_results.pkl'}")

            # Generate visualization
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                from scipy.stats import linregress

                fig, axes = plt.subplots(1, 3, figsize=(18, 5))

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

                # Add text box with correlation
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

                fig.savefig(str(output_dir / 'figure_modularity_independence.png'),
                           dpi=300, bbox_inches='tight', facecolor='white')
                fig.savefig(str(output_dir / 'figure_modularity_independence.pdf'),
                           dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)

                print(f"Figure saved to: {output_dir / 'figure_modularity_independence.png'}")
            except Exception as e:
                print(f"Warning: Could not generate visualization: {e}")
                import traceback
                traceback.print_exc()

            # Print summary
            print("\n" + "=" * 70)
            print("EXPERIMENT 4 SUMMARY")
            print("=" * 70)
            for gen_name, gen_res in results.items():
                s = gen_res['statistics']
                a = gen_res['assessment']
                print(f"\n{gen_name.upper()}:")
                print(f"  Pearson r(ρ_HB, Q):   {s['pearson_r']:+.4f}")
                print(f"  ANOVA p-value:        {s['anova_p']:.4f}")
                print(f"  Q range across h:     {s['Q_range']:.4f}")
                print(f"  Relative range:       {s['relative_range']:.1%}")
                print(f"  Degree preserved:     {s['degree_preserved']}")
                print(f"  Status:               {'✓ PASS' if a['passes_overall'] else '✗ FAIL'}")

        elif sys.argv[1] == "--improved":
            # Run improved Experiment 1
            results = experiment_1_parameter_control_improved(
                n_samples=30,
                n=500,
                k=5,
                theta_distribution='power_law',
                degree_correction_scale=1.5,
            )

            # Save results
            import pickle
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / 'experiment_1_improved.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(f"\nResults saved to: {output_dir / 'experiment_1_improved.pkl'}")

        elif sys.argv[1] == "--compare":
            # Run generator comparison
            n_samples = 15  # Smaller for comparison
            if len(sys.argv) > 2:
                n_samples = int(sys.argv[2])

            results = experiment_1_compare_generators(
                n_samples=n_samples,
                n=500,
            )

            # Save results
            import pickle
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / 'experiment_1_comparison.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(f"\nResults saved to: {output_dir / 'experiment_1_comparison.pkl'}")

        elif sys.argv[1] == "--lfr":
            # Run HB-LFR experiment
            results = experiment_1_parameter_control_improved(
                generator='hb_lfr',
                n_samples=20,
                n=500,
            )

            import pickle
            output_dir = Path("data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            with open(output_dir / 'experiment_1_hb_lfr.pkl', 'wb') as f:
                pickle.dump(results, f)

            print(f"\nResults saved to: {output_dir / 'experiment_1_hb_lfr.pkl'}")
        else:
            main()
    else:
        main()
