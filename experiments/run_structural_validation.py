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
    experiment_3_modularity_independence,
    experiment_4_concentration,
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


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--improved":
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
