#!/usr/bin/env python3
"""
Algorithmic Validation Runner
=============================

This script runs Experiments 7-8 for algorithmic validation of
hub-bridging generators.

Experiments:
7. Community detection algorithm performance
8. Sparsification algorithm behavior

Usage:
    python run_algorithmic_validation.py [--config CONFIG] [--output OUTPUT]
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_validation_config, load_network_params
from src.generators import hb_lfr
from src.validation import experiment_7_community_detection, experiment_8_sparsification
from src.visualization import (
    plot_algorithm_performance,
    plot_sparsification_effect,
    save_figure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run algorithmic validation experiments (7-8)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/algorithmic",
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
        "--experiments",
        type=str,
        nargs="+",
        choices=["7", "8", "all"],
        default=["all"],
        help="Which experiments to run",
    )
    return parser.parse_args()


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
        n_samples = config["sample_sizes"]["community_detection"]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator_params = network_params["lfr"]["default"].copy()
    generator_params.pop("h", None)

    experiments_to_run = ["7", "8"] if "all" in args.experiments else args.experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}

    # Experiment 7: Community Detection
    if "7" in experiments_to_run:
        logger.info("Running Experiment 7: Community Detection")

        exp_config = config["experiments"]["community_detection"]
        algorithms = exp_config.get("algorithms", ["louvain", "leiden"])

        results_7 = experiment_7_community_detection(
            generator_func=hb_lfr,
            generator_params=generator_params,
            algorithms=algorithms,
            n_samples=n_samples,
            seed=args.seed,
        )

        results["experiment_7"] = results_7

        json_path = output_dir / f"exp7_community_detection_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(_make_serializable(results_7), f, indent=2)
        logger.info(f"Saved Experiment 7 results to {json_path}")

        # Generate plots
        try:
            for metric in ["nmi", "ari"]:
                fig = plot_algorithm_performance(results_7, metric=metric)
                fig_path = output_dir / f"exp7_algorithm_{metric}.pdf"
                save_figure(fig, str(fig_path))
        except Exception as e:
            logger.warning(f"Could not create algorithm performance plot: {e}")

    # Experiment 8: Sparsification
    if "8" in experiments_to_run:
        logger.info("Running Experiment 8: Sparsification")

        exp_config = config["experiments"]["sparsification"]
        methods = exp_config.get("methods", ["dspar", "random"])
        ratios = exp_config.get("sparsification_ratios", [0.2, 0.3, 0.5])

        results_8 = experiment_8_sparsification(
            generator_func=hb_lfr,
            generator_params=generator_params,
            sparsification_methods=methods,
            sparsification_ratios=ratios,
            n_samples=n_samples,
            seed=args.seed,
        )

        results["experiment_8"] = results_8

        json_path = output_dir / f"exp8_sparsification_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(_make_serializable(results_8), f, indent=2)
        logger.info(f"Saved Experiment 8 results to {json_path}")

        # Generate plots
        try:
            for method in methods:
                fig = plot_sparsification_effect(results_8, method=method)
                fig_path = output_dir / f"exp8_sparsification_{method}.pdf"
                save_figure(fig, str(fig_path))
        except Exception as e:
            logger.warning(f"Could not create sparsification plot: {e}")

    # Summary
    logger.info("=" * 60)
    logger.info("ALGORITHMIC VALIDATION SUMMARY")
    logger.info("=" * 60)

    if "experiment_7" in results:
        logger.info("\nExperiment 7: Community Detection")
        summary = results["experiment_7"]["performance_summary"]
        for alg in results["experiment_7"]["algorithms"]:
            h0_nmi = summary[alg][0.0]["nmi_mean"]
            h1_nmi = summary[alg].get(1.0, summary[alg].get(max(summary[alg].keys())))["nmi_mean"]
            logger.info(f"  {alg}: NMI at h=0: {h0_nmi:.3f}, h=max: {h1_nmi:.3f}")

    if "experiment_8" in results:
        logger.info("\nExperiment 8: Sparsification")
        for method in results["experiment_8"]["methods"]:
            logger.info(f"  Method: {method}")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
