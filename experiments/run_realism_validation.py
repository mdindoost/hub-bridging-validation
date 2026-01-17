#!/usr/bin/env python3
"""
Realism Validation Runner
=========================

This script runs Experiments 5-6 for realism validation of
hub-bridging generators.

Experiments:
5. Property matching (synthetic vs real networks)
6. Network fitting (optimize parameters to match real)

Usage:
    python run_realism_validation.py [--config CONFIG] [--output OUTPUT]
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
from src.visualization import plot_property_comparison, save_figure

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


def main():
    """Main entry point."""
    args = parse_args()

    config = load_validation_config()
    network_params = load_network_params()

    if args.quick:
        n_samples = 5
        network_params["lfr"]["default"]["n"] = 250
    else:
        n_samples = config["sample_sizes"]["property_matching"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real networks
    real_networks = load_real_networks()

    # Set up generator
    generator_params = network_params["lfr"]["default"].copy()
    generator_params.pop("h", None)

    # Experiment 5: Property Matching
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
