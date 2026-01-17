#!/usr/bin/env python3
"""
Full Validation Runner
======================

This script runs all validation experiments (1-8) for hub-bridging
generators and produces a comprehensive validation report.

Usage:
    python run_full_validation.py [--config CONFIG] [--output OUTPUT]
    hb-validate  # If installed via setup.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run full validation suite for hub-bridging generators"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/full_validation",
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
        "--skip-realism",
        action="store_true",
        help="Skip realism experiments (5-6) which require real networks",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    logger.info("=" * 70)
    logger.info("HUB-BRIDGING GENERATOR VALIDATION FRAMEWORK")
    logger.info("=" * 70)
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info("")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {"timestamp": timestamp, "seed": args.seed, "quick_mode": args.quick}

    # Import and run structural validation
    logger.info("-" * 70)
    logger.info("PHASE 1: STRUCTURAL VALIDATION (Experiments 1-4)")
    logger.info("-" * 70)

    try:
        from run_structural_validation import (
            run_experiment_1,
            run_experiment_2,
            run_experiment_3,
            run_experiment_4,
            get_generator,
        )

        config = load_validation_config()
        network_params = load_network_params()

        if args.quick:
            n_samples = 5
            network_params["lfr"]["default"]["n"] = 250
        else:
            n_samples = None

        generator_params = network_params["lfr"]["default"].copy()
        generator_params.pop("h", None)
        generator_func = get_generator("hb_lfr")

        # Run structural experiments
        structural_results = {}

        logger.info("\n[1/4] Parameter Control...")
        structural_results["experiment_1"] = run_experiment_1(
            generator_func, generator_params, config, args.seed, n_samples
        )

        logger.info("\n[2/4] Degree Preservation...")
        structural_results["experiment_2"] = run_experiment_2(
            generator_func, generator_params, config, args.seed, n_samples
        )

        logger.info("\n[3/4] Modularity Independence...")
        structural_results["experiment_3"] = run_experiment_3(
            generator_func, generator_params, config, args.seed, n_samples
        )

        logger.info("\n[4/4] Concentration...")
        structural_results["experiment_4"] = run_experiment_4(
            generator_func, generator_params, config, args.seed, n_samples
        )

        all_results["structural"] = _summarize_structural(structural_results)

    except Exception as e:
        logger.error(f"Structural validation failed: {e}")
        all_results["structural"] = {"error": str(e)}

    # Phase 2: Realism validation
    if not args.skip_realism:
        logger.info("\n" + "-" * 70)
        logger.info("PHASE 2: REALISM VALIDATION (Experiments 5-6)")
        logger.info("-" * 70)

        try:
            # Import modules
            import networkx as nx
            from src.generators import hb_lfr
            from src.validation import experiment_5_property_matching

            # Use placeholder networks
            G = nx.karate_club_graph()
            communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
            real_networks = {"karate_club": (G, communities)}

            logger.info("\n[5/8] Property Matching...")
            results_5 = experiment_5_property_matching(
                real_networks=real_networks,
                generator_func=hb_lfr,
                generator_params=generator_params,
                n_samples=5 if args.quick else 10,
                seed=args.seed,
            )

            all_results["realism"] = {
                "property_matching": {
                    name: {"best_h": best["best_h"], "distance": best["distance"]}
                    for name, best in results_5.get("best_h_per_network", {}).items()
                }
            }

        except Exception as e:
            logger.error(f"Realism validation failed: {e}")
            all_results["realism"] = {"error": str(e)}
    else:
        logger.info("\n[Skipping realism validation as requested]")
        all_results["realism"] = {"skipped": True}

    # Phase 3: Algorithmic validation
    logger.info("\n" + "-" * 70)
    logger.info("PHASE 3: ALGORITHMIC VALIDATION (Experiments 7-8)")
    logger.info("-" * 70)

    try:
        from src.generators import hb_lfr
        from src.validation import experiment_7_community_detection, experiment_8_sparsification

        logger.info("\n[7/8] Community Detection...")
        results_7 = experiment_7_community_detection(
            generator_func=hb_lfr,
            generator_params=generator_params,
            algorithms=["louvain", "label_propagation"],
            n_samples=5 if args.quick else 20,
            seed=args.seed,
        )

        logger.info("\n[8/8] Sparsification...")
        results_8 = experiment_8_sparsification(
            generator_func=hb_lfr,
            generator_params=generator_params,
            sparsification_methods=["dspar", "random"],
            sparsification_ratios=[0.3, 0.5],
            n_samples=5 if args.quick else 20,
            seed=args.seed,
        )

        all_results["algorithmic"] = {
            "community_detection": _summarize_community_detection(results_7),
            "sparsification": "completed",
        }

    except Exception as e:
        logger.error(f"Algorithmic validation failed: {e}")
        all_results["algorithmic"] = {"error": str(e)}

    # Save full results
    results_path = output_dir / f"full_validation_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)

    # Print final summary
    _print_summary(all_results)

    logger.info(f"\nFull results saved to: {results_path}")


def _summarize_structural(results):
    """Summarize structural validation results."""
    summary = {}

    if "experiment_1" in results:
        res = results["experiment_1"]
        mono = res.get("monotonicity_test", {})
        summary["parameter_control"] = {
            "is_monotonic": mono.get("is_monotonic"),
            "spearman_r": mono.get("spearman_r"),
        }

    if "experiment_2" in results:
        res = results["experiment_2"]
        summary["degree_preservation"] = {
            "passed": res.get("preservation_passed"),
        }

    if "experiment_3" in results:
        res = results["experiment_3"]
        summary["modularity_independence"] = {
            "all_independent": res.get("all_independent"),
        }

    if "experiment_4" in results:
        res = results["experiment_4"]
        summary["concentration"] = {
            "all_concentrated": res.get("all_concentrated"),
        }

    return summary


def _summarize_community_detection(results):
    """Summarize community detection results."""
    summary = {}
    for alg in results.get("algorithms", []):
        h_values = results.get("h_values", [])
        if h_values:
            perf_0 = results["performance_summary"][alg].get(h_values[0], {})
            perf_max = results["performance_summary"][alg].get(h_values[-1], {})
            summary[alg] = {
                "nmi_at_h0": perf_0.get("nmi_mean"),
                "nmi_at_hmax": perf_max.get("nmi_mean"),
            }
    return summary


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
    elif isinstance(obj, bool):
        return bool(obj)
    elif callable(obj):
        return "<function>"
    else:
        return obj


def _print_summary(results):
    """Print final validation summary."""
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)

    # Structural
    struct = results.get("structural", {})
    if "error" not in struct:
        logger.info("\nStructural Validation:")
        pc = struct.get("parameter_control", {})
        logger.info(f"  [1] Parameter Control: {'PASS' if pc.get('is_monotonic') else 'FAIL'}")
        logger.info(f"  [2] Degree Preservation: {'PASS' if struct.get('degree_preservation', {}).get('passed') else 'FAIL'}")
        logger.info(f"  [3] Modularity Independence: {'PASS' if struct.get('modularity_independence', {}).get('all_independent') else 'FAIL'}")
        logger.info(f"  [4] Concentration: {'PASS' if struct.get('concentration', {}).get('all_concentrated') else 'FAIL'}")
    else:
        logger.info(f"\nStructural Validation: ERROR - {struct['error']}")

    # Realism
    real = results.get("realism", {})
    if real.get("skipped"):
        logger.info("\nRealism Validation: SKIPPED")
    elif "error" not in real:
        logger.info("\nRealism Validation: COMPLETED")
    else:
        logger.info(f"\nRealism Validation: ERROR - {real['error']}")

    # Algorithmic
    algo = results.get("algorithmic", {})
    if "error" not in algo:
        logger.info("\nAlgorithmic Validation: COMPLETED")
    else:
        logger.info(f"\nAlgorithmic Validation: ERROR - {algo['error']}")


if __name__ == "__main__":
    main()
