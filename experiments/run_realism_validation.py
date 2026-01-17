#!/usr/bin/env python3
"""
Realism Validation Runner
=========================

This script runs Experiments 5-6 for realism validation of
hub-bridging generators.

Experiments:
5. Real network property matching (HB-LFR vs Standard LFR)
6. Network fitting (optimize parameters to match real)

Usage:
    python run_realism_validation.py [--config CONFIG] [--output OUTPUT]
    python run_realism_validation.py --exp5 [--quick]
    python run_realism_validation.py --exp5 --data-dir /path/to/networks
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_validation_config, load_network_params
from src.generators import hb_lfr
from src.validation import experiment_5_property_matching, experiment_6_fitting
from src.validation.realism import (
    experiment_5_real_network_matching,
    experiment_5_extended,
    summarize_experiment_5,
    summarize_experiment_5_extended,
    PRIORITY_PROPERTIES,
    RHO_REGIMES,
)
from src.data import (
    load_real_networks_from_snap,
    create_sample_networks,
    load_networks_for_experiment_5,
)
from src.visualization import plot_property_comparison, plot_experiment_5_results, save_figure

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run realism validation experiments (5-6)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/realism",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation with reduced samples",
    )
    parser.add_argument(
        "--exp5",
        action="store_true",
        help="Run Experiment 5: Real Network Property Matching with h fitting",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/real_networks",
        help="Directory containing real network data (for --exp5)",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample networks (karate club) instead of loading from data-dir",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of synthetic samples per network",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Use extended experiment 5 with weighted properties and regime analysis",
    )
    parser.add_argument(
        "--sort-by-size",
        action="store_true",
        default=True,
        help="Process networks from smallest to largest (default: True)",
    )
    parser.add_argument(
        "--networks",
        type=str,
        nargs="+",
        default=None,
        help="Only process specific networks (e.g., --networks com-DBLP com-Amazon)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download networks before running (uses scripts/download_snap_networks.py)",
    )
    return parser.parse_args()


def load_real_networks():
    """
    Load real networks for comparison.

    Returns placeholder networks - replace with actual data loading.
    """
    networks = {}

    # Use NetworkX built-in networks as examples
    # In practice, load your actual real network datasets

    # Karate club (small social network)
    G = nx.karate_club_graph()
    communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
    networks["karate_club"] = (G, communities)

    # Generate synthetic "real" network for testing
    # Replace with actual real network loading
    logger.warning(
        "Using placeholder networks. "
        "Replace with actual real network loading in production."
    )

    return networks


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
    elif callable(obj):
        return "<function>"
    else:
        return obj


def run_experiment_5_real_network_matching(args, output_dir):
    """
    Run Experiment 5: Real Network Property Matching with h fitting.

    This validates that HB-LFR with fitted h parameter matches real
    network properties better than standard LFR.

    Supports two modes:
    - Standard: experiment_5_real_network_matching
    - Extended: experiment_5_extended (with weighted properties and regime analysis)
    """
    import pickle

    use_extended = getattr(args, 'extended', False)

    logger.info("=" * 70)
    if use_extended:
        logger.info("EXPERIMENT 5 (EXTENDED): Real Network Property Matching")
        logger.info("Features: Weighted properties, extended h range, regime analysis")
    else:
        logger.info("EXPERIMENT 5: Real Network Property Matching")
    logger.info("=" * 70)
    logger.info("Validating that HB-LFR with fitted h matches real networks")
    logger.info("better than standard LFR (h=0)")
    logger.info("")

    # Download networks if requested
    if getattr(args, 'download', False):
        logger.info("Downloading networks...")
        import subprocess
        result = subprocess.run(
            ['python', 'scripts/download_snap_networks.py', '--quick'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.warning(f"Download failed: {result.stderr}")
        else:
            logger.info("Download complete")

    # Load networks
    if args.use_sample:
        logger.info("Using sample networks (karate club)")
        networks = create_sample_networks()
    else:
        logger.info(f"Loading networks from: {args.data_dir}")
        if use_extended:
            # Use the enhanced loader for extended experiment
            networks = load_networks_for_experiment_5(
                data_dir=args.data_dir,
                min_nodes=100 if not args.quick else 30,
                max_nodes=100000 if not args.quick else 1000,
            )
        else:
            networks = load_real_networks_from_snap(
                data_dir=args.data_dir,
                min_nodes=30,
                max_nodes=5000 if not args.quick else 500,
            )

    if not networks:
        logger.warning("No networks found. Using karate club as fallback.")
        networks = create_sample_networks()

    logger.info(f"Loaded {len(networks)} networks")

    # Filter to specific networks if requested
    if getattr(args, 'networks', None):
        networks = {name: data for name, data in networks.items()
                   if any(n in name for n in args.networks)}
        logger.info(f"Filtered to {len(networks)} networks: {list(networks.keys())}")

    # Sort networks by size (smallest to largest) if requested
    if getattr(args, 'sort_by_size', True):
        networks = dict(sorted(
            networks.items(),
            key=lambda x: x[1]["G"].number_of_nodes()
        ))
        logger.info("Networks sorted by size (smallest to largest)")

    for name, data in networks.items():
        G = data["G"]
        expected_rho = data.get("metadata", {}).get("expected_rho_HB", "unknown")
        logger.info(f"  {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, expected ρ_HB={expected_rho}")

    # Set parameters based on quick mode
    if args.quick:
        n_synthetic_samples = 5
        n_calibration_samples = 5
        n_h_points = 15
    else:
        n_synthetic_samples = args.n_samples or 30
        n_calibration_samples = 10
        n_h_points = 25

    # Run experiment
    logger.info("\nRunning experiment...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if use_extended:
        results = experiment_5_extended(
            real_networks_dict=networks,
            n_synthetic_per_real=n_synthetic_samples,
            use_extended_h_fitting=True,
            use_weighted_distance=True,
            n_calibration_samples=n_calibration_samples,
            n_h_points=n_h_points,
            save_results=False,  # We'll save manually
            seed=args.seed,
        )
        summary = summarize_experiment_5_extended(results)
        results["summary"] = summary
        results_key = "networks"
    else:
        results = experiment_5_real_network_matching(
            real_networks_dict=networks,
            n_synthetic_per_real=n_synthetic_samples,
            n_calibration_samples=n_calibration_samples,
            seed=args.seed,
        )
        summary = summarize_experiment_5(results)
        results["summary"] = summary
        results_key = None  # Results are keyed by network name directly

    # Save results
    suffix = "_extended" if use_extended else ""

    # Save as pickle (preserves all data)
    pkl_path = output_dir / f"exp5_real_network_matching{suffix}_{timestamp}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)
    logger.info(f"Saved results to: {pkl_path}")

    # Save as JSON (human readable)
    json_path = output_dir / f"exp5_real_network_matching{suffix}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(results), f, indent=2)
    logger.info(f"Saved JSON to: {json_path}")

    # Generate visualization
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # For extended mode, convert to expected format for visualization
        if use_extended:
            viz_results = results  # Extended results already have 'networks' key
        else:
            viz_results = results

        fig = plot_experiment_5_results(
            viz_results,
            save_path=str(output_dir / f"figure_exp5_real_network_matching{suffix}.png")
        )
        plt.close(fig)
        logger.info(f"Saved figure to: {output_dir / f'figure_exp5_real_network_matching{suffix}.png'}")
    except Exception as e:
        logger.warning(f"Could not generate visualization: {e}")
        import traceback
        traceback.print_exc()

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 5 RESULTS")
    logger.info("=" * 70)

    logger.info("\nPer-Network Results:")
    logger.info("-" * 70)
    logger.info(f"{'Network':<20} {'Regime':<15} {'Fitted h':>10} {'HB-LFR Dist':>12} {'LFR Dist':>12} {'Improv%':>10}")
    logger.info("-" * 70)

    # Get networks dict based on mode
    if use_extended:
        networks_results = results.get("networks", {})
    else:
        networks_results = {k: v for k, v in results.items() if k not in ("summary", "metadata", "by_regime")}

    for name, net_res in networks_results.items():
        h = net_res.get("h_fitted", np.nan)
        hb_dist = net_res.get("overall_distance_hb", np.nan)
        lfr_dist = net_res.get("overall_distance_std", np.nan)
        regime = net_res.get("regime", "unknown")

        if not np.isnan(hb_dist) and not np.isnan(lfr_dist) and lfr_dist > 0:
            improvement = (lfr_dist - hb_dist) / lfr_dist * 100
        else:
            improvement = net_res.get("overall_improvement", np.nan)
            if not np.isnan(improvement):
                improvement *= 100

        logger.info(f"{name:<20} {regime:<15} {h:>10.3f} {hb_dist:>12.4f} {lfr_dist:>12.4f} {improvement:>+10.1f}")

    # Regime summary (for extended mode)
    if use_extended and "by_regime" in summary:
        logger.info("\n" + "-" * 70)
        logger.info("By ρ_HB Regime:")
        for regime_name, regime_data in summary.get("by_regime", {}).items():
            n = regime_data.get("n_networks", 0)
            wins = regime_data.get("hb_wins", 0)
            mean_imp = regime_data.get("mean_improvement", 0) * 100
            logger.info(f"  {regime_name}: {wins}/{n} wins, mean improvement: {mean_imp:+.1f}%")

    logger.info("")
    logger.info("Statistical Test (Mann-Whitney U):")
    stat_test = summary.get("statistical_test", {})
    logger.info(f"  U-statistic: {stat_test.get('U_statistic', np.nan):.2f}")
    logger.info(f"  p-value: {stat_test.get('p_value', np.nan):.4f}")
    logger.info(f"  Effect size: {stat_test.get('effect_size', np.nan):.3f}")

    logger.info("")
    logger.info(f"Average improvement: {summary.get('avg_improvement_percent', np.nan):.1f}%")
    logger.info(f"Networks where HB-LFR wins: {summary.get('hb_wins', 0)}/{summary.get('n_networks', 0)}")

    if use_extended:
        logger.info(f"Networks with achievable target: {summary.get('n_achievable', 0)}/{summary.get('n_networks', 0)}")

    passes = summary.get("passes", False)
    status = "PASS ✓" if passes else "FAIL ✗"
    logger.info(f"\nVALIDATION: {status}")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Experiment 5 (Real Network Matching) if requested
    if args.exp5:
        run_experiment_5_real_network_matching(args, output_dir)
        return

    # Original behavior: run both experiments 5 and 6
    config = load_validation_config()
    network_params = load_network_params()

    if args.quick:
        n_samples = 5
        network_params["lfr"]["default"]["n"] = 250
    else:
        n_samples = config["sample_sizes"]["property_matching"]

    # Load real networks
    real_networks = load_real_networks()

    # Set up generator
    generator_params = network_params["lfr"]["default"].copy()
    generator_params.pop("h", None)

    # Experiment 5: Property Matching (original version)
    logger.info("Running Experiment 5: Property Matching")
    results_5 = experiment_5_property_matching(
        real_networks=real_networks,
        generator_func=hb_lfr,
        generator_params=generator_params,
        n_samples=n_samples,
        seed=args.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"exp5_property_matching_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(_make_serializable(results_5), f, indent=2)
    logger.info(f"Saved Experiment 5 results to {json_path}")

    # Generate plots
    for prop in ["hub_bridging_ratio", "modularity"]:
        try:
            fig = plot_property_comparison(
                results_5["real_properties"],
                results_5["avg_synthetic_properties"],
                property_name=prop,
            )
            fig_path = output_dir / f"exp5_{prop}_comparison.pdf"
            save_figure(fig, str(fig_path))
        except Exception as e:
            logger.warning(f"Could not create plot for {prop}: {e}")

    # Experiment 6: Fitting (on first real network)
    if real_networks:
        first_network = list(real_networks.values())[0]
        logger.info("Running Experiment 6: Network Fitting")

        results_6 = experiment_6_fitting(
            target_network=first_network,
            generator_func=hb_lfr,
            fixed_params=generator_params,
            max_iterations=20 if args.quick else 50,
            seed=args.seed,
        )

        json_path = output_dir / f"exp6_fitting_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(_make_serializable(results_6), f, indent=2)
        logger.info(f"Saved Experiment 6 results to {json_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("REALISM VALIDATION SUMMARY")
    logger.info("=" * 60)

    logger.info("\nExperiment 5: Property Matching")
    for name, best in results_5.get("best_h_per_network", {}).items():
        logger.info(f"  {name}: best h = {best['best_h']:.2f}, distance = {best['distance']:.4f}")

    if "results_6" in dir():
        logger.info("\nExperiment 6: Network Fitting")
        logger.info(f"  Optimal params: {results_6.get('optimal_params', {})}")
        logger.info(f"  Optimal distance: {results_6.get('optimal_distance', np.nan):.4f}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
