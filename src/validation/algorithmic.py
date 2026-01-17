"""
Algorithmic Validation Module
=============================

This module implements Experiments 7-8 for validating algorithmic
behavior on hub-bridging networks:

- Experiment 7: Community detection algorithm performance
- Experiment 8: Sparsification algorithm behavior

References
----------
.. [1] Your PhD thesis or publication
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def experiment_7_community_detection(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    algorithms: Optional[List[str]] = None,
    h_values: Optional[List[float]] = None,
    n_samples: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 7: Analyze community detection performance vs h.

    This experiment tests whether hub-bridging affects the ability
    of community detection algorithms to recover the ground-truth
    communities.

    Parameters
    ----------
    generator_func : Callable
        Generator function that produces networks with communities
    generator_params : Dict[str, Any]
        Base generator parameters
    algorithms : List[str], optional
        Algorithms to test. Default: ['louvain', 'leiden', 'label_propagation']
    h_values : List[float], optional
        Values of h to test. Default: [0.0, 0.25, 0.5, 0.75, 1.0]
    n_samples : int, optional
        Number of samples per (algorithm, h) combination
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Any]
        Results with:
        - 'h_values': tested h values
        - 'algorithms': tested algorithms
        - 'nmi_scores': NMI scores [algorithm][h][samples]
        - 'ari_scores': ARI scores [algorithm][h][samples]
        - 'modularity_scores': detected modularity
        - 'performance_summary': mean/std for each (algorithm, h)
    """
    from ..algorithms.community_detection import (
        detect_communities,
        compute_nmi,
        compute_ari,
    )

    if algorithms is None:
        algorithms = ["louvain", "leiden", "label_propagation"]
    if h_values is None:
        h_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 7: Testing {len(algorithms)} algorithms "
        f"on {len(h_values)} h values"
    )

    # Initialize result storage
    nmi_scores: Dict[str, Dict[float, List[float]]] = {
        alg: {h: [] for h in h_values} for alg in algorithms
    }
    ari_scores: Dict[str, Dict[float, List[float]]] = {
        alg: {h: [] for h in h_values} for alg in algorithms
    }
    modularity_scores: Dict[str, Dict[float, List[float]]] = {
        alg: {h: [] for h in h_values} for alg in algorithms
    }

    for h in h_values:
        logger.info(f"  h = {h}")

        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                # Generate network
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, true_communities = generator_func(**params)

                # Test each algorithm
                for alg in algorithms:
                    try:
                        detected = detect_communities(G, algorithm=alg)

                        nmi = compute_nmi(true_communities, detected, list(G.nodes()))
                        ari = compute_ari(true_communities, detected, list(G.nodes()))

                        # Compute modularity of detected partition
                        community_sets = {}
                        for node, comm in detected.items():
                            if comm not in community_sets:
                                community_sets[comm] = set()
                            community_sets[comm].add(node)
                        Q = nx.community.modularity(G, list(community_sets.values()))

                        nmi_scores[alg][h].append(nmi)
                        ari_scores[alg][h].append(ari)
                        modularity_scores[alg][h].append(Q)

                    except Exception as e:
                        logger.debug(f"Algorithm {alg} failed: {e}")

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")

    # Compute summary statistics
    performance_summary = {}
    for alg in algorithms:
        performance_summary[alg] = {}
        for h in h_values:
            nmi_vals = nmi_scores[alg][h]
            ari_vals = ari_scores[alg][h]

            performance_summary[alg][h] = {
                "nmi_mean": np.mean(nmi_vals) if nmi_vals else np.nan,
                "nmi_std": np.std(nmi_vals) if nmi_vals else np.nan,
                "ari_mean": np.mean(ari_vals) if ari_vals else np.nan,
                "ari_std": np.std(ari_vals) if ari_vals else np.nan,
                "n_samples": len(nmi_vals),
            }

    results = {
        "h_values": h_values,
        "algorithms": algorithms,
        "nmi_scores": nmi_scores,
        "ari_scores": ari_scores,
        "modularity_scores": modularity_scores,
        "performance_summary": performance_summary,
    }

    logger.info("Experiment 7 complete")
    return results


def experiment_8_sparsification(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    sparsification_methods: Optional[List[str]] = None,
    sparsification_ratios: Optional[List[float]] = None,
    h_values: Optional[List[float]] = None,
    n_samples: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 8: Analyze sparsification behavior vs h.

    This experiment tests whether hub-bridging affects the behavior
    of graph sparsification algorithms, particularly DSpar.

    Parameters
    ----------
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Base generator parameters
    sparsification_methods : List[str], optional
        Methods to test. Default: ['dspar', 'random', 'degree_based']
    sparsification_ratios : List[float], optional
        Edge retention ratios. Default: [0.1, 0.2, 0.3, 0.5]
    h_values : List[float], optional
        Values of h to test. Default: [0.0, 0.5, 1.0]
    n_samples : int, optional
        Number of samples per combination
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Any]
        Results with:
        - 'h_values': tested h values
        - 'methods': tested methods
        - 'ratios': tested ratios
        - 'community_preservation': how well communities preserved after sparsification
        - 'property_preservation': network properties after sparsification
        - 'edge_type_retention': inter vs intra edge retention rates
    """
    from ..algorithms.sparsification import sparsify_graph
    from ..metrics.hub_bridging import classify_edges_by_hub_bridging
    from ..algorithms.community_detection import detect_communities, compute_nmi

    if sparsification_methods is None:
        sparsification_methods = ["dspar", "random", "degree_based"]
    if sparsification_ratios is None:
        sparsification_ratios = [0.1, 0.2, 0.3, 0.5]
    if h_values is None:
        h_values = [0.0, 0.5, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 8: Testing {len(sparsification_methods)} methods "
        f"at {len(sparsification_ratios)} ratios, {len(h_values)} h values"
    )

    # Initialize result storage
    results_data: Dict[str, Dict[float, Dict[float, Dict]]] = {
        method: {
            ratio: {h: {"community_nmi": [], "inter_retention": [], "intra_retention": []}
                    for h in h_values}
            for ratio in sparsification_ratios
        }
        for method in sparsification_methods
    }

    for h in h_values:
        logger.info(f"  h = {h}")

        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                # Generate network
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, communities = generator_func(**params)

                # Classify original edges
                inter_edges, intra_edges = classify_edges_by_hub_bridging(G, communities)
                n_inter_original = len(inter_edges)
                n_intra_original = len(intra_edges)

                # Test each method and ratio
                for method in sparsification_methods:
                    for ratio in sparsification_ratios:
                        try:
                            # Sparsify
                            G_sparse = sparsify_graph(
                                G, method=method, ratio=ratio, seed=sample_seed
                            )

                            # Compute community detection NMI on sparse graph
                            detected = detect_communities(G_sparse, algorithm="louvain")
                            nmi = compute_nmi(communities, detected, list(G_sparse.nodes()))

                            # Count retained edge types
                            inter_sparse, intra_sparse = classify_edges_by_hub_bridging(
                                G_sparse, communities
                            )
                            inter_retention = len(inter_sparse) / max(n_inter_original, 1)
                            intra_retention = len(intra_sparse) / max(n_intra_original, 1)

                            results_data[method][ratio][h]["community_nmi"].append(nmi)
                            results_data[method][ratio][h]["inter_retention"].append(inter_retention)
                            results_data[method][ratio][h]["intra_retention"].append(intra_retention)

                        except Exception as e:
                            logger.debug(f"Sparsification failed: {e}")

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")

    # Compute summary statistics
    summary = {}
    for method in sparsification_methods:
        summary[method] = {}
        for ratio in sparsification_ratios:
            summary[method][ratio] = {}
            for h in h_values:
                data = results_data[method][ratio][h]
                summary[method][ratio][h] = {
                    "community_nmi_mean": np.mean(data["community_nmi"]) if data["community_nmi"] else np.nan,
                    "community_nmi_std": np.std(data["community_nmi"]) if data["community_nmi"] else np.nan,
                    "inter_retention_mean": np.mean(data["inter_retention"]) if data["inter_retention"] else np.nan,
                    "intra_retention_mean": np.mean(data["intra_retention"]) if data["intra_retention"] else np.nan,
                    "n_samples": len(data["community_nmi"]),
                }

    results = {
        "h_values": h_values,
        "methods": sparsification_methods,
        "ratios": sparsification_ratios,
        "detailed_results": results_data,
        "summary": summary,
    }

    logger.info("Experiment 8 complete")
    return results


def analyze_dspar_effectiveness(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    n_samples: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Analyze how effectively DSpar separates inter/intra edges.

    This is a focused analysis on DSpar's ability to correctly
    identify and prioritize inter-community edges for removal.

    Parameters
    ----------
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Base generator parameters
    h_values : List[float], optional
        Values of h to test
    n_samples : int, optional
        Number of samples per h
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Any]
        Analysis results including ROC-AUC for DSpar scores
    """
    from ..metrics.hub_bridging import compute_dspar_score, classify_edges_by_hub_bridging
    from sklearn.metrics import roc_auc_score

    if h_values is None:
        h_values = [0.0, 0.5, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(f"Analyzing DSpar effectiveness for {len(h_values)} h values")

    auc_scores: Dict[float, List[float]] = {h: [] for h in h_values}
    separation_scores: Dict[float, List[float]] = {h: [] for h in h_values}

    for h in h_values:
        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, communities = generator_func(**params)
                degrees = dict(G.degree())

                # Get edge labels and DSpar scores
                inter_edges, intra_edges = classify_edges_by_hub_bridging(G, communities)

                labels = []
                scores = []

                for u, v in inter_edges:
                    labels.append(1)  # Inter = positive class
                    scores.append(compute_dspar_score(degrees[u], degrees[v]))

                for u, v in intra_edges:
                    labels.append(0)  # Intra = negative class
                    scores.append(compute_dspar_score(degrees[u], degrees[v]))

                if len(set(labels)) == 2:  # Need both classes for AUC
                    auc = roc_auc_score(labels, scores)
                    auc_scores[h].append(auc)

                # Compute separation (difference of means)
                inter_scores = [scores[i] for i in range(len(labels)) if labels[i] == 1]
                intra_scores = [scores[i] for i in range(len(labels)) if labels[i] == 0]
                if inter_scores and intra_scores:
                    separation = (np.mean(inter_scores) - np.mean(intra_scores)) / (
                        np.std(inter_scores + intra_scores) + 1e-10
                    )
                    separation_scores[h].append(separation)

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")

    results = {
        "h_values": h_values,
        "auc_scores": auc_scores,
        "separation_scores": separation_scores,
        "summary": {
            h: {
                "auc_mean": np.mean(auc_scores[h]) if auc_scores[h] else np.nan,
                "auc_std": np.std(auc_scores[h]) if auc_scores[h] else np.nan,
                "separation_mean": np.mean(separation_scores[h]) if separation_scores[h] else np.nan,
            }
            for h in h_values
        },
    }

    logger.info("DSpar effectiveness analysis complete")
    return results
