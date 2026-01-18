"""
Realism Validation Module
=========================

This module implements Experiments 5-6 for validating that
hub-bridging generators can match properties of real networks:

- Experiment 5: Property matching (compare synthetic vs real)
- Experiment 6: Network fitting (optimize parameters to match real)

Features:
- Incremental result saving (saves after each network to prevent data loss)
- CSV export for easy analysis in spreadsheets
- Pickle export for full data preservation

References
----------
.. [1] Your PhD thesis or publication

Community Detection:
    Uses Leiden algorithm (Traag et al. 2019) for all networks.
    Random seed fixed at 42 for reproducibility.
    Leiden is superior to Louvain (fixes disconnected communities).
"""

import csv
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from ..config import RANDOM_SEED, COMMUNITY_DETECTION_SEED

logger = logging.getLogger(__name__)


# Priority properties for Experiment 5 (extended version)
# Ordered by importance for hub-bridging validation
PRIORITY_PROPERTIES = [
    'rho_HB',              # CRITICAL - primary validation metric
    'delta_DSpar',         # VERY IMPORTANT - connects to previous paper
    'modularity',          # Important for community structure
    'degree_assortativity',  # Important structural property
    'clustering_avg',      # Standard benchmark property
    'power_law_alpha',     # Degree distribution
    'transitivity',        # Alternative clustering measure
    'avg_path_length',     # Distance property
    'rich_club_10',        # Hub connectivity
]

# Weights for overall distance calculation
# Higher weight = more important for matching
PROPERTY_WEIGHTS = {
    'rho_HB': 3.0,              # Triple weight - primary metric
    'delta_DSpar': 2.0,         # Double weight - important for thesis
    'modularity': 1.5,
    'degree_assortativity': 1.0,
    'clustering_avg': 1.0,
    'power_law_alpha': 1.0,
    'transitivity': 0.5,
    'avg_path_length': 0.5,
    'rich_club_10': 0.5,
}

# ρ_HB regimes for categorization
RHO_REGIMES = {
    'extreme_hub_bridging': (4.0, float('inf')),   # ρ > 4 (e.g., wiki-Talk)
    'strong_hub_bridging': (2.0, 4.0),             # 2 < ρ ≤ 4
    'moderate_hub_bridging': (1.0, 2.0),           # 1 < ρ ≤ 2
    'hub_neutral': (0.8, 1.0),                     # 0.8 < ρ ≤ 1
    'hub_isolation': (0.0, 0.8),                   # ρ ≤ 0.8 (e.g., ca-GrQc)
}


# =============================================================================
# Incremental Saving and CSV Export Utilities
# =============================================================================

def _get_results_file_paths(
    results_dir: str,
    prefix: str = "exp5",
    timestamp: Optional[str] = None,
    network_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Get file paths for results storage.

    Parameters
    ----------
    results_dir : str
        Directory for results
    prefix : str
        File prefix (default: "exp5")
    timestamp : str, optional
        Timestamp string (default: generates new one)
    network_name : str, optional
        If provided, include in filename (for parallel execution)

    Returns
    -------
    dict
        Dictionary with paths for pickle, csv, and incremental files
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Include network name if provided (for parallel execution)
    if network_name:
        suffix = f"{timestamp}_{network_name}"
    else:
        suffix = timestamp

    return {
        'pickle': results_path / f"{prefix}_{suffix}.pkl",
        'csv': results_path / f"{prefix}_{suffix}.csv",
        'incremental': results_path / f"{prefix}_{suffix}_incremental.pkl",
        'timestamp': timestamp,
    }


def _save_incremental_result(
    filepath: Path,
    network_name: str,
    network_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save result incrementally after processing each network.

    This ensures no data is lost if the experiment is interrupted.

    Parameters
    ----------
    filepath : Path
        Path to incremental pickle file
    network_name : str
        Name of the network just processed
    network_result : dict
        Results for this network
    metadata : dict, optional
        Experiment metadata
    """
    # Load existing results if file exists
    if filepath.exists():
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    else:
        data = {
            'networks': {},
            'metadata': metadata or {},
            'last_updated': None,
        }

    # Add new network result
    data['networks'][network_name] = network_result
    data['last_updated'] = datetime.now().isoformat()

    # Save updated results
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    logger.debug(f"Incremental save: {network_name} -> {filepath}")


def _save_results_to_csv(
    filepath: Path,
    results: Dict[str, Any],
    properties: Optional[List[str]] = None,
) -> None:
    """
    Save experiment 5 results to CSV format.

    Creates a CSV with one row per network containing:
    - Network name and metadata
    - Fitted h value
    - HB-LFR and Standard LFR distances
    - Improvement metrics
    - Per-property comparisons

    Parameters
    ----------
    filepath : Path
        Path to CSV file
    results : dict
        Results from experiment_5 functions
    properties : list, optional
        Properties to include in CSV
    """
    if properties is None:
        properties = PRIORITY_PROPERTIES

    # Determine if this is extended format or standard format
    if 'networks' in results:
        networks_data = results['networks']
    else:
        # Filter out non-network keys
        networks_data = {k: v for k, v in results.items()
                        if k not in ('summary', 'metadata', 'by_regime')}

    if not networks_data:
        logger.warning("No network results to save to CSV")
        return

    # Build CSV header
    header = [
        'network_name',
        'domain',
        'n_nodes',
        'n_edges',
        'regime',
        'rho_HB_real',
        'h_fitted',
        'achievable',
        'hb_distance',
        'std_distance',
        'improvement',
        'improvement_percent',
        'n_hb_generated',
        'n_std_generated',
    ]

    # Add property-specific columns
    for prop in properties:
        header.extend([
            f'{prop}_real',
            f'{prop}_hb_mean',
            f'{prop}_std_mean',
            f'{prop}_improvement',
        ])

    # Write CSV
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for net_name, net_data in networks_data.items():
            row = [net_name]

            # Basic info
            real_props = net_data.get('real_properties', {})
            lfr_params = net_data.get('lfr_params', {})

            row.append(net_data.get('domain', 'unknown'))
            row.append(lfr_params.get('n', real_props.get('n', '')))
            row.append(real_props.get('m', ''))
            row.append(net_data.get('regime', 'unknown'))
            row.append(real_props.get('rho_HB', ''))

            # Fitted h
            row.append(net_data.get('h_fitted', ''))
            fit_result = net_data.get('fit_result', {})
            row.append(fit_result.get('achievable', True))

            # Distances
            hb_dist = net_data.get('overall_distance_hb', np.nan)
            std_dist = net_data.get('overall_distance_std', np.nan)
            improvement = net_data.get('overall_improvement', np.nan)

            row.append(hb_dist if not np.isnan(hb_dist) else '')
            row.append(std_dist if not np.isnan(std_dist) else '')
            row.append(improvement if not np.isnan(improvement) else '')
            row.append(improvement * 100 if not np.isnan(improvement) else '')

            # Counts
            row.append(net_data.get('n_hb_generated', ''))
            row.append(net_data.get('n_std_generated', ''))

            # Property-specific data
            hb_mean_props = net_data.get('hb_mean_properties', {})
            std_mean_props = net_data.get('std_mean_properties', {})
            comparison = net_data.get('comparison', {})

            for prop in properties:
                real_val = real_props.get(prop, np.nan)
                row.append(real_val if not np.isnan(real_val) else '')

                hb_val = hb_mean_props.get(prop, np.nan)
                row.append(hb_val if not np.isnan(hb_val) else '')

                std_val = std_mean_props.get(prop, np.nan)
                row.append(std_val if not np.isnan(std_val) else '')

                # Property improvement
                if prop in comparison:
                    prop_imp = comparison[prop].get('improvement', np.nan)
                    row.append(prop_imp if not np.isnan(prop_imp) else '')
                else:
                    row.append('')

            writer.writerow(row)

    logger.info(f"Saved CSV results to {filepath}")


def _save_summary_to_csv(
    filepath: Path,
    summary: Dict[str, Any],
) -> None:
    """
    Save experiment summary to a separate CSV.

    Parameters
    ----------
    filepath : Path
        Path to CSV file (will append _summary before extension)
    summary : dict
        Summary from summarize_experiment_5 functions
    """
    summary_path = filepath.parent / f"{filepath.stem}_summary.csv"

    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])

        # Overall metrics
        writer.writerow(['n_networks', summary.get('n_networks', 0)])
        writer.writerow(['hb_wins', summary.get('hb_wins', 0)])
        writer.writerow(['win_rate', summary.get('win_rate', 0)])
        writer.writerow(['mean_improvement', summary.get('mean_improvement', 0)])
        writer.writerow(['avg_improvement_percent', summary.get('avg_improvement_percent', 0)])
        writer.writerow(['median_improvement', summary.get('median_improvement', 0)])

        # Statistical test
        stat_test = summary.get('statistical_test', {})
        writer.writerow(['U_statistic', stat_test.get('U_statistic', '')])
        writer.writerow(['p_value', stat_test.get('p_value', '')])
        writer.writerow(['effect_size', stat_test.get('effect_size', '')])

        writer.writerow(['passes', summary.get('passes', False)])

        # Regime breakdown
        writer.writerow([])
        writer.writerow(['regime', 'n_networks', 'hb_wins', 'mean_improvement'])
        for regime, regime_data in summary.get('by_regime', {}).items():
            writer.writerow([
                regime,
                regime_data.get('n_networks', 0),
                regime_data.get('hb_wins', 0),
                regime_data.get('mean_improvement', 0),
            ])

    logger.info(f"Saved summary CSV to {summary_path}")


def experiment_5_property_matching(
    real_networks: Dict[str, Tuple[nx.Graph, Dict[int, int]]],
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    n_samples: int = 20,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 5: Compare synthetic network properties to real networks.

    This experiment generates synthetic networks at various h values
    and compares their properties to a set of real networks.

    Parameters
    ----------
    real_networks : Dict[str, Tuple[nx.Graph, Dict[int, int]]]
        Dictionary mapping network name to (graph, communities)
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Base generator parameters
    h_values : List[float], optional
        Values of h to test. Default: [0.0, 0.25, 0.5, 0.75, 1.0]
    n_samples : int, optional
        Number of synthetic samples per h value (default: 20)
    seed : int, optional
        Base random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary with comparison results:
        - 'real_properties': properties of each real network
        - 'synthetic_properties': properties of synthetic networks per h
        - 'distance_matrix': distances between real and synthetic
        - 'best_h_per_network': h value that best matches each real network
        - 'property_comparison': detailed property-by-property comparison
    """
    from ..metrics.network_properties import comprehensive_network_properties
    from ..metrics.distance_metrics import property_distance_vector

    if h_values is None:
        h_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 5: Comparing {len(real_networks)} real networks "
        f"with synthetic at {len(h_values)} h values"
    )

    # Compute properties for real networks
    real_properties = {}
    for name, (G, communities) in real_networks.items():
        logger.info(f"  Computing properties for real network: {name}")
        real_properties[name] = comprehensive_network_properties(
            G, communities, compute_expensive=True
        )

    # Generate synthetic networks and compute properties
    synthetic_properties: Dict[float, List[Dict]] = {h: [] for h in h_values}

    for h in h_values:
        logger.info(f"  Generating synthetic networks at h={h}")

        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, communities = generator_func(**params)
                props = comprehensive_network_properties(
                    G, communities, compute_expensive=True
                )
                synthetic_properties[h].append(props)

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")

    # Compute average synthetic properties for each h
    avg_synthetic_properties = {}
    for h, props_list in synthetic_properties.items():
        if props_list:
            avg_synthetic_properties[h] = _average_properties(props_list)

    # Compute distance matrix (real network x h value)
    distance_matrix = {}
    for name, real_props in real_properties.items():
        distance_matrix[name] = {}
        for h, synth_props in avg_synthetic_properties.items():
            distances = property_distance_vector(real_props, synth_props)
            distance_matrix[name][h] = distances["weighted_total"]

    # Find best h for each real network
    best_h_per_network = {}
    for name, h_distances in distance_matrix.items():
        if h_distances:
            best_h = min(h_distances.keys(), key=lambda h: h_distances[h])
            best_h_per_network[name] = {
                "best_h": best_h,
                "distance": h_distances[best_h],
            }

    # Detailed property comparison
    property_comparison = _detailed_property_comparison(
        real_properties, avg_synthetic_properties
    )

    results = {
        "h_values": h_values,
        "real_networks": list(real_networks.keys()),
        "real_properties": real_properties,
        "avg_synthetic_properties": avg_synthetic_properties,
        "distance_matrix": distance_matrix,
        "best_h_per_network": best_h_per_network,
        "property_comparison": property_comparison,
    }

    logger.info("Experiment 5 complete")
    return results


def _average_properties(props_list: List[Dict]) -> Dict:
    """Average properties from multiple samples."""
    if not props_list:
        return {}

    avg = {}

    # Average basic properties
    if "basic" in props_list[0]:
        avg["basic"] = {
            key: np.mean([p["basic"].get(key, np.nan) for p in props_list])
            for key in props_list[0]["basic"]
            if isinstance(props_list[0]["basic"].get(key), (int, float))
        }

    # Average degree properties
    if "degree" in props_list[0]:
        avg["degree"] = {
            key: np.mean([p["degree"].get(key, np.nan) for p in props_list])
            for key in ["mean", "std", "max", "skewness", "gini"]
            if key in props_list[0]["degree"]
        }

    # Average clustering
    if "clustering" in props_list[0]:
        avg["clustering"] = {
            key: np.mean([p["clustering"].get(key, np.nan) for p in props_list])
            for key in props_list[0]["clustering"]
        }

    # Average path length
    if "path_length" in props_list[0]:
        avg["path_length"] = {
            key: np.mean([p["path_length"].get(key, np.nan) for p in props_list])
            for key in props_list[0]["path_length"]
        }

    # Average community properties
    if "community" in props_list[0]:
        avg["community"] = {
            "modularity": np.mean([
                p["community"].get("modularity", np.nan) for p in props_list
            ]),
        }

    # Average hub-bridging
    if "hub_bridging" in props_list[0]:
        avg["hub_bridging"] = {
            "rho_hb": np.mean([
                p["hub_bridging"].get("rho_hb", np.nan) for p in props_list
            ]),
            "delta_dspar": np.mean([
                p["hub_bridging"].get("delta_dspar", np.nan) for p in props_list
            ]),
        }

    return avg


def _detailed_property_comparison(
    real_properties: Dict[str, Dict],
    synthetic_properties: Dict[float, Dict],
) -> Dict[str, Any]:
    """Create detailed property-by-property comparison."""
    comparison = {}

    # Compare hub-bridging ratio
    comparison["hub_bridging_ratio"] = {
        "real": {
            name: props.get("hub_bridging", {}).get("rho_hb", np.nan)
            for name, props in real_properties.items()
        },
        "synthetic": {
            h: props.get("hub_bridging", {}).get("rho_hb", np.nan)
            for h, props in synthetic_properties.items()
        },
    }

    # Compare modularity
    comparison["modularity"] = {
        "real": {
            name: props.get("community", {}).get("modularity", np.nan)
            for name, props in real_properties.items()
        },
        "synthetic": {
            h: props.get("community", {}).get("modularity", np.nan)
            for h, props in synthetic_properties.items()
        },
    }

    # Compare clustering
    comparison["clustering"] = {
        "real": {
            name: props.get("clustering", {}).get("global", np.nan)
            for name, props in real_properties.items()
        },
        "synthetic": {
            h: props.get("clustering", {}).get("global", np.nan)
            for h, props in synthetic_properties.items()
        },
    }

    return comparison


def experiment_6_fitting(
    target_network: Tuple[nx.Graph, Dict[int, int]],
    generator_func: Callable,
    fixed_params: Optional[Dict[str, Any]] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    n_samples_per_eval: int = 5,
    max_iterations: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 6: Fit generator parameters to match a real network.

    This experiment uses optimization to find generator parameters
    (especially h) that minimize the distance to a target real network.

    Parameters
    ----------
    target_network : Tuple[nx.Graph, Dict[int, int]]
        Target network to match (graph, communities)
    generator_func : Callable
        Generator function
    fixed_params : Dict[str, Any], optional
        Parameters to keep fixed (e.g., 'n' = target n)
    param_bounds : Dict[str, Tuple[float, float]], optional
        Bounds for parameters to optimize.
        Default: {'h': (0, 2), 'mu': (0.05, 0.7)}
    n_samples_per_eval : int, optional
        Samples to average per evaluation (default: 5)
    max_iterations : int, optional
        Maximum optimization iterations (default: 50)
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Any]
        Fitting results:
        - 'optimal_params': best parameters found
        - 'optimal_distance': distance at optimal params
        - 'optimization_history': history of evaluations
        - 'target_properties': properties of target network
        - 'fitted_properties': properties at optimal params
    """
    from ..metrics.network_properties import comprehensive_network_properties
    from ..metrics.distance_metrics import property_distance_vector

    G_target, communities_target = target_network
    n_target = G_target.number_of_nodes()

    rng = np.random.default_rng(seed)

    logger.info(f"Experiment 6: Fitting generator to network with {n_target} nodes")

    # Compute target properties
    target_properties = comprehensive_network_properties(
        G_target, communities_target, compute_expensive=True
    )

    # Set up parameters
    if fixed_params is None:
        fixed_params = {"n": n_target}
    else:
        fixed_params = fixed_params.copy()
        fixed_params["n"] = n_target

    if param_bounds is None:
        param_bounds = {
            "h": (0.0, 2.0),
            "mu": (0.05, 0.7),
        }

    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    optimization_history = []

    def objective(x):
        """Objective function to minimize."""
        params = fixed_params.copy()
        for i, name in enumerate(param_names):
            params[name] = x[i]

        # Generate samples and compute average distance
        distances = []
        for j in range(n_samples_per_eval):
            try:
                params["seed"] = int(rng.integers(0, 2**31))
                G, communities = generator_func(**params)
                props = comprehensive_network_properties(
                    G, communities, compute_expensive=False
                )
                dist = property_distance_vector(target_properties, props)
                distances.append(dist["weighted_total"])
            except Exception as e:
                logger.debug(f"Evaluation failed: {e}")
                distances.append(float("inf"))

        mean_distance = np.mean(distances)

        optimization_history.append({
            "params": {name: x[i] for i, name in enumerate(param_names)},
            "distance": mean_distance,
        })

        return mean_distance

    # Run optimization
    logger.info("Starting optimization...")
    result = optimize.differential_evolution(
        objective,
        bounds,
        maxiter=max_iterations,
        seed=seed,
        disp=True,
        polish=True,
    )

    # Extract optimal parameters
    optimal_params = fixed_params.copy()
    for i, name in enumerate(param_names):
        optimal_params[name] = result.x[i]

    # Generate networks at optimal params and compute properties
    fitted_properties_list = []
    for j in range(10):
        try:
            optimal_params["seed"] = int(rng.integers(0, 2**31))
            G, communities = generator_func(**optimal_params)
            props = comprehensive_network_properties(
                G, communities, compute_expensive=True
            )
            fitted_properties_list.append(props)
        except Exception:
            pass

    fitted_properties = _average_properties(fitted_properties_list) if fitted_properties_list else {}

    results = {
        "optimal_params": {
            name: optimal_params[name]
            for name in param_names
        },
        "optimal_distance": result.fun,
        "optimization_success": result.success,
        "n_evaluations": result.nfev,
        "optimization_history": optimization_history,
        "target_properties": target_properties,
        "fitted_properties": fitted_properties,
        "fixed_params": fixed_params,
    }

    logger.info(
        f"Experiment 6 complete: optimal h={optimal_params.get('h', 'N/A'):.4f}, "
        f"distance={result.fun:.6f}"
    )

    return results


def batch_property_matching(
    real_networks: Dict[str, Tuple[nx.Graph, Dict[int, int]]],
    generators: Dict[str, Tuple[Callable, Dict[str, Any]]],
    h_values: Optional[List[float]] = None,
    n_samples: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run property matching for multiple generators.

    Parameters
    ----------
    real_networks : Dict[str, Tuple[nx.Graph, Dict[int, int]]]
        Real networks to compare against
    generators : Dict[str, Tuple[Callable, Dict[str, Any]]]
        Named generators with their base parameters
    h_values : List[float], optional
        Values of h to test
    n_samples : int, optional
        Samples per (generator, h) combination
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Any]
        Comparison results for all generator-network pairs
    """
    results = {}

    for gen_name, (gen_func, gen_params) in generators.items():
        logger.info(f"Testing generator: {gen_name}")
        results[gen_name] = experiment_5_property_matching(
            real_networks=real_networks,
            generator_func=gen_func,
            generator_params=gen_params,
            h_values=h_values,
            n_samples=n_samples,
            seed=seed,
        )

    return results


def experiment_5_real_network_matching(
    real_networks_dict: Dict[str, Dict[str, Any]],
    n_synthetic_per_real: int = 50,
    properties_to_compare: Optional[List[str]] = None,
    n_calibration_samples: int = 10,
    n_h_points: int = 15,
    save_results: bool = True,
    results_dir: str = "data/results",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Experiment 5: Validate that HB-LFR matches real networks better than standard LFR.

    For each real network:
        1. Measure properties
        2. Extract LFR parameters
        3. Fit h to match rho_HB
        4. Generate HB-LFR networks (fitted h)
        5. Generate Standard LFR networks (h=0)
        6. Compare properties: Real vs HB-LFR vs Standard LFR
        7. Statistical tests

    Parameters
    ----------
    real_networks_dict : dict
        From load_real_networks_from_snap() with structure:
        {
            'network_name': {
                'G': nx.Graph,
                'domain': str,
                'communities': list of sets,
                'metadata': dict
            }
        }
    n_synthetic_per_real : int
        Number of synthetic networks to generate per real network
    properties_to_compare : list of str
        Properties to compare. If None, use default comprehensive set.
    n_calibration_samples : int
        Samples per h value during calibration
    n_h_points : int
        Number of h values to test during calibration
    save_results : bool
        Whether to save results to disk
    results_dir : str
        Directory to save results
    seed : int
        Random seed

    Returns
    -------
    dict
        Comprehensive results for each network
    """
    from scipy.stats import mannwhitneyu

    from ..generators.calibration import extract_lfr_params_from_real, fit_h_to_real_network
    from ..generators.hb_lfr import hb_lfr
    from ..metrics.network_properties import comprehensive_network_properties_flat

    rng = np.random.default_rng(seed)

    if properties_to_compare is None:
        properties_to_compare = [
            'rho_HB', 'delta_DSpar', 'degree_assortativity',
            'clustering_avg', 'modularity', 'rich_club_10',
            'avg_path_length', 'power_law_alpha', 'transitivity'
        ]

    logger.info("=" * 70)
    logger.info("EXPERIMENT 5: Real Network Property Matching")
    logger.info("=" * 70)
    logger.info(f"Networks to process: {len(real_networks_dict)}")
    logger.info(f"Synthetic samples per network: {n_synthetic_per_real}")
    logger.info(f"Properties to compare: {properties_to_compare}")
    logger.info("")

    # Set up file paths for incremental saving
    file_paths = _get_results_file_paths(results_dir, prefix="exp5_real_network_matching")
    logger.info(f"Results will be saved to: {file_paths['pickle']}")
    logger.info(f"Incremental saves to: {file_paths['incremental']}")

    results = {}
    metadata = {
        'n_synthetic_per_real': n_synthetic_per_real,
        'properties': properties_to_compare,
        'seed': seed,
        'timestamp': file_paths['timestamp'],
    }

    for net_name, net_data in real_networks_dict.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {net_name}")
        logger.info(f"{'=' * 60}")

        G_real = net_data['G']
        communities_real = net_data['communities']

        # Step 1: Measure real network properties
        logger.info("Step 1: Measuring real network properties...")
        real_props = comprehensive_network_properties_flat(
            G_real, communities_real, net_name
        )

        # Step 2: Extract LFR parameters
        logger.info("Step 2: Extracting LFR parameters...")
        try:
            lfr_params = extract_lfr_params_from_real(G_real, communities_real)
        except Exception as e:
            logger.error(f"Failed to extract LFR params for {net_name}: {e}")
            continue

        # Step 3: Fit h parameter
        logger.info("Step 3: Fitting h parameter...")
        try:
            fit_result = fit_h_to_real_network(
                G_real, communities_real, lfr_params,
                n_calibration_samples=n_calibration_samples,
                n_h_points=n_h_points,
                seed=int(rng.integers(0, 2**31))
            )
            h_fitted = fit_result['h_fitted']
            logger.info(f"  Fitted h = {h_fitted:.3f} (target rho = {fit_result['rho_target']:.3f})")
        except Exception as e:
            logger.error(f"Failed to fit h for {net_name}: {e}")
            # Use default h=1.0
            h_fitted = 1.0
            fit_result = {'h_fitted': h_fitted, 'rho_target': np.nan}
            logger.warning(f"  Using default h = {h_fitted}")

        # Step 4: Generate HB-LFR networks
        # Use reduced max_iters if target was not achievable (saves time)
        target_achievable = fit_result.get('achievable', True)
        generation_max_iters = 500 if not target_achievable else 5000

        if not target_achievable:
            logger.info(f"Step 4: Generating {n_synthetic_per_real} HB-LFR networks (h={h_fitted:.3f}, FAST MODE - target not achievable)...")
        else:
            logger.info(f"Step 4: Generating {n_synthetic_per_real} HB-LFR networks (h={h_fitted:.3f})...")

        hb_lfr_props = []

        for i in range(n_synthetic_per_real):
            sample_seed = int(rng.integers(0, 2**31))
            try:
                G_hb, communities_hb = hb_lfr(
                    n=lfr_params['n'],
                    tau1=lfr_params['tau1'],
                    tau2=lfr_params.get('tau2', 1.5),
                    mu=lfr_params['mu'],
                    average_degree=lfr_params.get('average_degree'),
                    min_community=lfr_params.get('min_community'),
                    max_community=lfr_params.get('max_community'),
                    h=h_fitted,
                    seed=sample_seed,
                    max_iters=generation_max_iters,
                )
                props = comprehensive_network_properties_flat(
                    G_hb, communities_hb, f"{net_name}_HB_{i}"
                )
                hb_lfr_props.append(props)
            except Exception as e:
                logger.debug(f"  HB-LFR generation {i} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{n_synthetic_per_real} HB-LFR networks")

        logger.info(f"  Successfully generated {len(hb_lfr_props)} HB-LFR networks")

        # Step 5: Generate Standard LFR networks (h=0)
        logger.info(f"Step 5: Generating {n_synthetic_per_real} Standard LFR networks (h=0)...")
        std_lfr_props = []

        for i in range(n_synthetic_per_real):
            sample_seed = int(rng.integers(0, 2**31))
            try:
                G_std, communities_std = hb_lfr(
                    n=lfr_params['n'],
                    tau1=lfr_params['tau1'],
                    tau2=lfr_params.get('tau2', 1.5),
                    mu=lfr_params['mu'],
                    average_degree=lfr_params.get('average_degree'),
                    min_community=lfr_params.get('min_community'),
                    max_community=lfr_params.get('max_community'),
                    h=0.0,
                    seed=sample_seed,
                )
                props = comprehensive_network_properties_flat(
                    G_std, communities_std, f"{net_name}_STD_{i}"
                )
                std_lfr_props.append(props)
            except Exception as e:
                logger.debug(f"  Standard LFR generation {i} failed: {e}")
                continue

            if (i + 1) % 10 == 0:
                logger.info(f"  Generated {i + 1}/{n_synthetic_per_real} Standard LFR networks")

        logger.info(f"  Successfully generated {len(std_lfr_props)} Standard LFR networks")

        # Step 6: Compare properties
        logger.info("Step 6: Computing property comparisons...")
        comparison = {}

        for prop in properties_to_compare:
            real_val = real_props.get(prop, np.nan)

            if np.isnan(real_val):
                logger.debug(f"  Property {prop} not available for {net_name}")
                continue

            # Extract values from synthetic networks
            hb_vals = [p.get(prop, np.nan) for p in hb_lfr_props]
            hb_vals = [v for v in hb_vals if not np.isnan(v)]

            std_vals = [p.get(prop, np.nan) for p in std_lfr_props]
            std_vals = [v for v in std_vals if not np.isnan(v)]

            if len(hb_vals) < 5 or len(std_vals) < 5:
                logger.debug(f"  Insufficient samples for {prop}")
                continue

            hb_mean = np.mean(hb_vals)
            hb_std = np.std(hb_vals)
            std_mean = np.mean(std_vals)
            std_std = np.std(std_vals)

            # Distance from real
            hb_distance = abs(real_val - hb_mean)
            std_distance = abs(real_val - std_mean)

            # Improvement
            if std_distance > 0:
                improvement = (std_distance - hb_distance) / std_distance
            else:
                improvement = 0.0

            # Statistical test: are HB distances smaller than Standard distances?
            hb_distances = [abs(real_val - v) for v in hb_vals]
            std_distances = [abs(real_val - v) for v in std_vals]

            try:
                _, p_value = mannwhitneyu(hb_distances, std_distances, alternative='less')
            except Exception:
                p_value = np.nan

            comparison[prop] = {
                'real_value': real_val,
                'hb_mean': hb_mean,
                'hb_std': hb_std,
                'std_mean': std_mean,
                'std_std': std_std,
                'hb_distance': hb_distance,
                'std_distance': std_distance,
                'improvement': improvement,
                'p_value': p_value
            }

        # Overall distance (normalized)
        overall_hb_dist = 0.0
        overall_std_dist = 0.0
        n_props = 0

        for prop in properties_to_compare:
            if prop in comparison:
                c = comparison[prop]
                real_val = c['real_value']

                # Normalize by real value (relative error)
                if real_val != 0 and not np.isnan(real_val):
                    hb_norm_dist = c['hb_distance'] / abs(real_val)
                    std_norm_dist = c['std_distance'] / abs(real_val)

                    overall_hb_dist += hb_norm_dist ** 2
                    overall_std_dist += std_norm_dist ** 2
                    n_props += 1

        if n_props > 0:
            overall_hb_dist = np.sqrt(overall_hb_dist / n_props)
            overall_std_dist = np.sqrt(overall_std_dist / n_props)
            overall_improvement = (overall_std_dist - overall_hb_dist) / overall_std_dist if overall_std_dist > 0 else 0
        else:
            overall_hb_dist = np.nan
            overall_std_dist = np.nan
            overall_improvement = np.nan

        # Log summary
        logger.info(f"\n--- Results for {net_name} ---")
        logger.info(f"  h_fitted: {h_fitted:.3f}")
        logger.info(f"  Overall HB-LFR distance: {overall_hb_dist:.4f}")
        logger.info(f"  Overall Standard LFR distance: {overall_std_dist:.4f}")
        logger.info(f"  Improvement: {overall_improvement:.1%}")

        # Store results
        results[net_name] = {
            'real_properties': real_props,
            'domain': net_data.get('domain', 'unknown'),
            'lfr_params': lfr_params,
            'h_fitted': h_fitted,
            'fit_result': fit_result,
            'hb_lfr_properties': hb_lfr_props,
            'standard_lfr_properties': std_lfr_props,
            'comparison': comparison,
            'overall_distance_hb': overall_hb_dist,
            'overall_distance_std': overall_std_dist,
            'overall_improvement': overall_improvement,
            'n_hb_generated': len(hb_lfr_props),
            'n_std_generated': len(std_lfr_props),
        }

        # Save incrementally after each network (prevents data loss)
        if save_results:
            _save_incremental_result(
                filepath=file_paths['incremental'],
                network_name=net_name,
                network_result=results[net_name],
                metadata=metadata,
            )
            # Also update CSV after each network
            _save_results_to_csv(
                filepath=file_paths['csv'],
                results=results,
                properties=properties_to_compare,
            )
            logger.info(f"  Saved incremental results for {net_name}")

    # Save final results (pickle + CSV + summary)
    if save_results and results:
        # Save complete pickle
        with open(file_paths['pickle'], 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to {file_paths['pickle']}")

        # Generate and save summary
        summary = summarize_experiment_5(results)
        results['summary'] = summary
        _save_summary_to_csv(file_paths['csv'], summary)

        # Update pickle with summary
        with open(file_paths['pickle'], 'wb') as f:
            pickle.dump(results, f)

    return results


def summarize_experiment_5(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary statistics for Experiment 5.

    Parameters
    ----------
    results : dict
        Results from experiment_5_real_network_matching()

    Returns
    -------
    dict
        Summary statistics
    """
    if not results:
        return {
            'n_networks': 0,
            'properties_tested': [],
            'n_improved': 0,
            'mean_improvement': 0.0,
            'median_improvement': 0.0,
            'significant_improvements': 0,
        }

    # Collect all properties tested
    all_properties = set()
    for net_results in results.values():
        all_properties.update(net_results.get('comparison', {}).keys())

    properties_tested = sorted(list(all_properties))

    # Per-property summary
    per_property_summary = {}
    for prop in properties_tested:
        improvements = []
        p_values = []

        for net_name, net_results in results.items():
            if prop in net_results.get('comparison', {}):
                c = net_results['comparison'][prop]
                improvements.append(c['improvement'])
                if not np.isnan(c.get('p_value', np.nan)):
                    p_values.append(c['p_value'])

        if improvements:
            per_property_summary[prop] = {
                'mean_improvement': np.mean(improvements),
                'median_improvement': np.median(improvements),
                'std_improvement': np.std(improvements),
                'n_improved': sum(1 for i in improvements if i > 0),
                'n_networks': len(improvements),
                'n_significant': sum(1 for p in p_values if p < 0.05),
            }

    # Per-network summary
    per_network_summary = {}
    overall_improvements = []

    for net_name, net_results in results.items():
        improvement = net_results.get('overall_improvement', np.nan)
        if not np.isnan(improvement):
            overall_improvements.append(improvement)

        per_network_summary[net_name] = {
            'h_fitted': net_results.get('h_fitted', np.nan),
            'overall_improvement': improvement,
            'hb_distance': net_results.get('overall_distance_hb', np.nan),
            'std_distance': net_results.get('overall_distance_std', np.nan),
        }

    # Count significant improvements
    significant_count = 0
    for net_results in results.values():
        for prop_comp in net_results.get('comparison', {}).values():
            if prop_comp.get('p_value', 1.0) < 0.05 and prop_comp.get('improvement', 0) > 0:
                significant_count += 1

    # Compute statistical test (Mann-Whitney U) for overall distances
    hb_all_distances = []
    std_all_distances = []
    for net_results in results.values():
        hb_dist = net_results.get('overall_distance_hb', np.nan)
        std_dist = net_results.get('overall_distance_std', np.nan)
        if not np.isnan(hb_dist):
            hb_all_distances.append(hb_dist)
        if not np.isnan(std_dist):
            std_all_distances.append(std_dist)

    statistical_test = {}
    if len(hb_all_distances) >= 2 and len(std_all_distances) >= 2:
        try:
            from scipy.stats import mannwhitneyu
            U_stat, p_value = mannwhitneyu(hb_all_distances, std_all_distances, alternative='less')
            n1, n2 = len(hb_all_distances), len(std_all_distances)
            effect_size = 1 - (2 * U_stat) / (n1 * n2)  # rank-biserial correlation
            statistical_test = {
                'U_statistic': U_stat,
                'p_value': p_value,
                'effect_size': effect_size,
            }
        except Exception:
            statistical_test = {'U_statistic': np.nan, 'p_value': np.nan, 'effect_size': np.nan}
    else:
        statistical_test = {'U_statistic': np.nan, 'p_value': np.nan, 'effect_size': np.nan}

    # Count networks where HB-LFR wins
    hb_wins = sum(1 for i in overall_improvements if i > 0)

    # Determine if validation passes
    # Criteria: HB-LFR should have smaller distance in majority of networks
    # or statistically significant improvement
    passes = False
    if len(results) > 0:
        win_rate = hb_wins / len(results)
        p_val = statistical_test.get('p_value', 1.0)
        passes = (win_rate > 0.5) or (not np.isnan(p_val) and p_val < 0.05)

    summary = {
        'n_networks': len(results),
        'properties_tested': properties_tested,
        'n_improved': sum(1 for i in overall_improvements if i > 0),
        'hb_wins': hb_wins,
        'mean_improvement': np.mean(overall_improvements) if overall_improvements else 0.0,
        'avg_improvement_percent': np.mean(overall_improvements) * 100 if overall_improvements else 0.0,
        'median_improvement': np.median(overall_improvements) if overall_improvements else 0.0,
        'std_improvement': np.std(overall_improvements) if overall_improvements else 0.0,
        'significant_improvements': significant_count,
        'statistical_test': statistical_test,
        'passes': passes,
        'per_property_summary': per_property_summary,
        'per_network_summary': per_network_summary,
    }

    return summary


def categorize_by_rho_regime(rho: float) -> str:
    """Categorize a network by its ρ_HB regime."""
    for regime_name, (low, high) in RHO_REGIMES.items():
        if low < rho <= high:
            return regime_name
    return 'unknown'


def compute_weighted_distance(
    real_props: Dict[str, float],
    synth_props: Dict[str, float],
    properties: List[str],
    weights: Optional[Dict[str, float]] = None,
    max_relative_error: float = 2.0,
) -> float:
    """
    Compute weighted property distance between real and synthetic networks.

    Parameters
    ----------
    real_props : dict
        Real network properties
    synth_props : dict
        Synthetic network properties
    properties : list
        Properties to compare
    weights : dict, optional
        Property weights (default: PROPERTY_WEIGHTS)
    max_relative_error : float
        Maximum allowed relative error per property (default: 2.0 = 200%)
        This prevents properties with extreme differences from dominating.

    Returns
    -------
    float
        Weighted RMSE distance
    """
    if weights is None:
        weights = PROPERTY_WEIGHTS

    total_weight = 0.0
    weighted_squared_error = 0.0

    # Properties that can be negative or close to zero need special handling
    # Use absolute difference scaled to typical range instead of relative error
    special_properties = {
        'degree_assortativity': 2.0,   # Range is [-1, 1], so scale factor is 2
        'modularity': 1.0,              # Range is [0, 1]
        'transitivity': 1.0,            # Range is [0, 1]
        'clustering_avg': 1.0,          # Range is [0, 1]
        'delta_DSpar': 1.0,             # Can be close to 0, range ~[-0.5, 0.5]
    }

    for prop in properties:
        real_val = real_props.get(prop, np.nan)
        synth_val = synth_props.get(prop, np.nan)

        if np.isnan(real_val) or np.isnan(synth_val):
            continue

        weight = weights.get(prop, 1.0)

        # For properties with bounded ranges, use scaled absolute error
        if prop in special_properties:
            scale = special_properties[prop]
            error = abs(synth_val - real_val) / scale
        # For unbounded properties, use relative error with cap
        elif abs(real_val) > 1e-10:
            error = abs(synth_val - real_val) / abs(real_val)
            # Cap relative error to prevent extreme values from dominating
            error = min(error, max_relative_error)
        else:
            # real_val is essentially zero
            error = abs(synth_val - real_val)
            error = min(error, max_relative_error)

        weighted_squared_error += weight * (error ** 2)
        total_weight += weight

    if total_weight > 0:
        return np.sqrt(weighted_squared_error / total_weight)
    else:
        return np.nan


def experiment_5_extended(
    real_networks_dict: Dict[str, Dict[str, Any]],
    n_synthetic_per_real: int = 30,
    use_extended_h_fitting: bool = True,
    use_weighted_distance: bool = True,
    properties_to_compare: Optional[List[str]] = None,
    n_calibration_samples: int = 10,
    n_h_points: int = 25,
    save_results: bool = True,
    results_dir: str = "data/results",
    seed: int = 42,
    max_workers: int = 1,
) -> Dict[str, Any]:
    """
    Extended Experiment 5: Real Network Property Matching with weighted metrics.

    This extended version:
    - Uses extended h fitting for extreme ρ_HB values
    - Applies weighted property comparison (ρ_HB and δ get higher weights)
    - Categorizes networks by ρ_HB regime
    - Provides regime-specific analysis

    Parameters
    ----------
    real_networks_dict : dict
        From load_real_networks_from_snap() or load_networks_for_experiment_5()
    n_synthetic_per_real : int
        Synthetic networks per real network
    use_extended_h_fitting : bool
        Whether to use fit_h_to_real_network_extended (handles extreme ρ_HB)
    use_weighted_distance : bool
        Whether to use weighted property distance
    properties_to_compare : list, optional
        Properties to compare (default: PRIORITY_PROPERTIES)
    n_calibration_samples : int
        Samples per h value in calibration
    n_h_points : int
        Number of h values in calibration
    save_results : bool
        Whether to save results to disk
    results_dir : str
        Directory for results
    seed : int
        Random seed

    Returns
    -------
    dict
        Comprehensive results including regime-specific analysis
    """
    from scipy.stats import mannwhitneyu

    from ..generators.calibration import (
        extract_lfr_params_from_real,
        fit_h_to_real_network,
        fit_h_to_real_network_extended,
    )
    from ..generators.hb_lfr import hb_lfr
    from ..metrics.network_properties import comprehensive_network_properties_flat

    rng = np.random.default_rng(seed)

    if properties_to_compare is None:
        properties_to_compare = PRIORITY_PROPERTIES.copy()

    logger.info("=" * 70)
    logger.info("EXPERIMENT 5 (EXTENDED): Real Network Property Matching")
    logger.info("=" * 70)
    logger.info(f"Networks to process: {len(real_networks_dict)}")
    logger.info(f"Synthetic samples per network: {n_synthetic_per_real}")
    logger.info(f"Extended h fitting: {use_extended_h_fitting}")
    logger.info(f"Weighted distance: {use_weighted_distance}")
    logger.info(f"Properties to compare: {properties_to_compare}")
    logger.info("")

    # Set up file paths for incremental saving
    # Include network name in filename if single network (parallel execution mode)
    single_network_name = list(real_networks_dict.keys())[0] if len(real_networks_dict) == 1 else None
    file_paths = _get_results_file_paths(results_dir, prefix="exp5_extended", network_name=single_network_name)
    logger.info(f"Results will be saved to: {file_paths['pickle']}")
    logger.info(f"Incremental saves to: {file_paths['incremental']}")

    results = {
        'networks': {},
        'by_regime': {},
        'metadata': {
            'n_synthetic_per_real': n_synthetic_per_real,
            'use_extended_h_fitting': use_extended_h_fitting,
            'use_weighted_distance': use_weighted_distance,
            'properties': properties_to_compare,
            'seed': seed,
            'timestamp': file_paths['timestamp'],
        }
    }

    # Select fitting function
    fit_func = fit_h_to_real_network_extended if use_extended_h_fitting else fit_h_to_real_network

    for net_name, net_data in real_networks_dict.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {net_name}")
        logger.info(f"{'=' * 60}")

        G_real = net_data['G']
        communities_real = net_data['communities']

        # Step 1: Measure real network properties
        logger.info("Step 1: Measuring real network properties...")
        real_props = comprehensive_network_properties_flat(
            G_real, communities_real, net_name
        )

        # Categorize by regime
        rho_real = real_props.get('rho_HB', np.nan)
        regime = categorize_by_rho_regime(rho_real) if not np.isnan(rho_real) else 'unknown'
        logger.info(f"  ρ_HB = {rho_real:.3f} (regime: {regime})")

        # Step 2: Extract LFR parameters
        logger.info("Step 2: Extracting LFR parameters...")
        try:
            lfr_params = extract_lfr_params_from_real(G_real, communities_real)
        except Exception as e:
            logger.error(f"Failed to extract LFR params for {net_name}: {e}")
            continue

        # Step 3: Fit h parameter
        logger.info("Step 3: Fitting h parameter...")
        try:
            fit_result = fit_func(
                G_real, communities_real, lfr_params,
                n_calibration_samples=n_calibration_samples,
                n_h_points=n_h_points,
                seed=int(rng.integers(0, 2**31))
            )
            h_fitted = fit_result['h_fitted']
            achievable = fit_result.get('achievable', True)
            logger.info(f"  Fitted h = {h_fitted:.3f}")
            if not achievable:
                logger.info(f"  Note: Target ρ_HB was outside achievable range")
        except Exception as e:
            logger.error(f"Failed to fit h for {net_name}: {e}")
            h_fitted = 1.0
            fit_result = {'h_fitted': h_fitted, 'rho_target': rho_real, 'achievable': False}
            logger.warning(f"  Using default h = {h_fitted}")

        # Step 4: Generate HB-LFR networks
        # Use reduced max_iters if target was not achievable (saves time)
        target_achievable = fit_result.get('achievable', True)
        generation_max_iters = 500 if not target_achievable else 5000

        if not target_achievable:
            logger.info(f"Step 4: Generating {n_synthetic_per_real} HB-LFR networks (h={h_fitted:.3f}, FAST MODE - target not achievable)...")
        else:
            logger.info(f"Step 4: Generating {n_synthetic_per_real} HB-LFR networks (h={h_fitted:.3f})...")

        hb_lfr_props = []

        for i in range(n_synthetic_per_real):
            sample_seed = int(rng.integers(0, 2**31))
            try:
                G_hb, communities_hb = hb_lfr(
                    n=lfr_params['n'],
                    tau1=lfr_params['tau1'],
                    tau2=lfr_params.get('tau2', 1.5),
                    mu=lfr_params['mu'],
                    average_degree=lfr_params.get('average_degree'),
                    min_community=lfr_params.get('min_community'),
                    max_community=lfr_params.get('max_community'),
                    h=h_fitted,
                    seed=sample_seed,
                    max_iters=generation_max_iters,
                )
                props = comprehensive_network_properties_flat(
                    G_hb, communities_hb, f"{net_name}_HB_{i}"
                )
                hb_lfr_props.append(props)
            except Exception as e:
                logger.debug(f"  HB-LFR generation {i} failed: {e}")
                continue

        logger.info(f"  Successfully generated {len(hb_lfr_props)} HB-LFR networks")

        # Step 5: Generate Standard LFR networks (h=0)
        logger.info(f"Step 5: Generating {n_synthetic_per_real} Standard LFR networks (h=0)...")
        std_lfr_props = []

        for i in range(n_synthetic_per_real):
            sample_seed = int(rng.integers(0, 2**31))
            try:
                G_std, communities_std = hb_lfr(
                    n=lfr_params['n'],
                    tau1=lfr_params['tau1'],
                    tau2=lfr_params.get('tau2', 1.5),
                    mu=lfr_params['mu'],
                    average_degree=lfr_params.get('average_degree'),
                    min_community=lfr_params.get('min_community'),
                    max_community=lfr_params.get('max_community'),
                    h=0.0,
                    seed=sample_seed,
                )
                props = comprehensive_network_properties_flat(
                    G_std, communities_std, f"{net_name}_STD_{i}"
                )
                std_lfr_props.append(props)
            except Exception as e:
                logger.debug(f"  Standard LFR generation {i} failed: {e}")
                continue

        logger.info(f"  Successfully generated {len(std_lfr_props)} Standard LFR networks")

        # Step 6: Compute distances
        logger.info("Step 6: Computing property distances...")

        if len(hb_lfr_props) < 3 or len(std_lfr_props) < 3:
            logger.warning(f"  Insufficient samples for {net_name}, skipping")
            continue

        # Compute mean properties for synthetic networks
        hb_mean_props = {}
        std_mean_props = {}

        for prop in properties_to_compare:
            hb_vals = [p.get(prop, np.nan) for p in hb_lfr_props]
            hb_vals = [v for v in hb_vals if not np.isnan(v)]
            if hb_vals:
                hb_mean_props[prop] = np.mean(hb_vals)

            std_vals = [p.get(prop, np.nan) for p in std_lfr_props]
            std_vals = [v for v in std_vals if not np.isnan(v)]
            if std_vals:
                std_mean_props[prop] = np.mean(std_vals)

        # Compute weighted distances
        if use_weighted_distance:
            hb_distance = compute_weighted_distance(real_props, hb_mean_props, properties_to_compare)
            std_distance = compute_weighted_distance(real_props, std_mean_props, properties_to_compare)
        else:
            # Simple RMSE
            hb_distance = compute_weighted_distance(
                real_props, hb_mean_props, properties_to_compare,
                weights={p: 1.0 for p in properties_to_compare}
            )
            std_distance = compute_weighted_distance(
                real_props, std_mean_props, properties_to_compare,
                weights={p: 1.0 for p in properties_to_compare}
            )

        # Improvement
        if std_distance > 0 and not np.isnan(std_distance) and not np.isnan(hb_distance):
            improvement = (std_distance - hb_distance) / std_distance
        else:
            improvement = np.nan

        # Build per-property comparison dict for CSV export
        comparison = {}
        for prop in properties_to_compare:
            real_val = real_props.get(prop, np.nan)
            hb_val = hb_mean_props.get(prop, np.nan)
            std_val = std_mean_props.get(prop, np.nan)

            # Per-property improvement: how much closer is HB-LFR to real vs Standard LFR
            if not np.isnan(real_val) and real_val != 0:
                hb_err = abs(hb_val - real_val) / abs(real_val) if not np.isnan(hb_val) else np.nan
                std_err = abs(std_val - real_val) / abs(real_val) if not np.isnan(std_val) else np.nan
                if not np.isnan(hb_err) and not np.isnan(std_err) and std_err > 0:
                    prop_improvement = (std_err - hb_err) / std_err
                else:
                    prop_improvement = np.nan
            else:
                prop_improvement = np.nan

            comparison[prop] = {
                'real_value': real_val,
                'hb_mean': hb_val,
                'std_mean': std_val,
                'improvement': prop_improvement,
            }

        logger.info(f"  HB-LFR distance: {hb_distance:.4f}")
        logger.info(f"  Standard LFR distance: {std_distance:.4f}")
        logger.info(f"  Improvement: {improvement:.1%}" if not np.isnan(improvement) else "  Improvement: N/A")

        # Store network results
        results['networks'][net_name] = {
            'real_properties': real_props,
            'domain': net_data.get('domain', 'unknown'),
            'lfr_params': lfr_params,
            'h_fitted': h_fitted,
            'fit_result': fit_result,
            'regime': regime,
            'hb_lfr_properties': hb_lfr_props,
            'standard_lfr_properties': std_lfr_props,
            'hb_mean_properties': hb_mean_props,
            'std_mean_properties': std_mean_props,
            'comparison': comparison,
            'overall_distance_hb': hb_distance,
            'overall_distance_std': std_distance,
            'overall_improvement': improvement,
            'n_hb_generated': len(hb_lfr_props),
            'n_std_generated': len(std_lfr_props),
        }

        # Add to regime tracking
        if regime not in results['by_regime']:
            results['by_regime'][regime] = []
        results['by_regime'][regime].append(net_name)

        # Save incrementally after each network (prevents data loss)
        if save_results:
            _save_incremental_result(
                filepath=file_paths['incremental'],
                network_name=net_name,
                network_result=results['networks'][net_name],
                metadata=results['metadata'],
            )
            # Also update CSV after each network
            _save_results_to_csv(
                filepath=file_paths['csv'],
                results=results,
                properties=properties_to_compare,
            )
            logger.info(f"  Saved incremental results for {net_name}")

    # Save final results (pickle + CSV)
    if save_results and results['networks']:
        # Save complete pickle
        with open(file_paths['pickle'], 'wb') as f:
            pickle.dump(results, f)
        logger.info(f"\nResults saved to {file_paths['pickle']}")

        # Save CSV for easy analysis
        _save_results_to_csv(
            filepath=file_paths['csv'],
            results=results,
            properties=properties_to_compare,
        )

        # Generate and save summary
        summary = summarize_experiment_5_extended(results)
        results['summary'] = summary
        _save_summary_to_csv(file_paths['csv'], summary)

        # Update pickle with summary
        with open(file_paths['pickle'], 'wb') as f:
            pickle.dump(results, f)

    return results


def summarize_experiment_5_extended(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate extended summary statistics for Experiment 5.

    Provides analysis broken down by:
    - Overall statistics
    - ρ_HB regime
    - Property-specific improvements

    Parameters
    ----------
    results : dict
        Results from experiment_5_extended()

    Returns
    -------
    dict
        Comprehensive summary
    """
    from scipy.stats import mannwhitneyu

    networks = results.get('networks', {})
    if not networks:
        return {'n_networks': 0, 'passes': False}

    # Overall statistics
    hb_distances = []
    std_distances = []
    improvements = []
    achievable_count = 0

    for net_name, net_res in networks.items():
        hb_dist = net_res.get('overall_distance_hb', np.nan)
        std_dist = net_res.get('overall_distance_std', np.nan)
        imp = net_res.get('overall_improvement', np.nan)

        if not np.isnan(hb_dist):
            hb_distances.append(hb_dist)
        if not np.isnan(std_dist):
            std_distances.append(std_dist)
        if not np.isnan(imp):
            improvements.append(imp)

        fit_res = net_res.get('fit_result', {})
        if fit_res.get('achievable', True):
            achievable_count += 1

    # Statistical test
    statistical_test = {}
    if len(hb_distances) >= 2 and len(std_distances) >= 2:
        try:
            U_stat, p_value = mannwhitneyu(hb_distances, std_distances, alternative='less')
            n1, n2 = len(hb_distances), len(std_distances)
            effect_size = 1 - (2 * U_stat) / (n1 * n2)
            statistical_test = {
                'U_statistic': U_stat,
                'p_value': p_value,
                'effect_size': effect_size,
            }
        except Exception:
            statistical_test = {'U_statistic': np.nan, 'p_value': np.nan, 'effect_size': np.nan}

    # Regime-specific analysis
    regime_summary = {}
    for regime_name, regime_networks in results.get('by_regime', {}).items():
        regime_improvements = []
        regime_hb_wins = 0

        for net_name in regime_networks:
            net_res = networks.get(net_name, {})
            imp = net_res.get('overall_improvement', np.nan)
            if not np.isnan(imp):
                regime_improvements.append(imp)
                if imp > 0:
                    regime_hb_wins += 1

        if regime_improvements:
            regime_summary[regime_name] = {
                'n_networks': len(regime_networks),
                'networks': regime_networks,
                'mean_improvement': np.mean(regime_improvements),
                'median_improvement': np.median(regime_improvements),
                'hb_wins': regime_hb_wins,
                'win_rate': regime_hb_wins / len(regime_networks),
            }

    # Pass criteria
    hb_wins = sum(1 for imp in improvements if imp > 0)
    n_networks = len(networks)
    win_rate = hb_wins / n_networks if n_networks > 0 else 0
    p_val = statistical_test.get('p_value', 1.0)

    passes = (win_rate > 0.5) or (not np.isnan(p_val) and p_val < 0.05)

    summary = {
        'n_networks': n_networks,
        'n_achievable': achievable_count,
        'hb_wins': hb_wins,
        'win_rate': win_rate,
        'mean_improvement': np.mean(improvements) if improvements else 0.0,
        'avg_improvement_percent': np.mean(improvements) * 100 if improvements else 0.0,
        'median_improvement': np.median(improvements) if improvements else 0.0,
        'std_improvement': np.std(improvements) if improvements else 0.0,
        'mean_hb_distance': np.mean(hb_distances) if hb_distances else np.nan,
        'mean_std_distance': np.mean(std_distances) if std_distances else np.nan,
        'statistical_test': statistical_test,
        'passes': passes,
        'by_regime': regime_summary,
    }

    return summary
