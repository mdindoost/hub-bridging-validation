"""
Realism Validation Module
=========================

This module implements Experiments 5-6 for validating that
hub-bridging generators can match properties of real networks:

- Experiment 5: Property matching (compare synthetic vs real)
- Experiment 6: Network fitting (optimize parameters to match real)

References
----------
.. [1] Your PhD thesis or publication
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import optimize

logger = logging.getLogger(__name__)


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
