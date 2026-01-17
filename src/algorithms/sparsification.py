"""
Sparsification Module
=====================

This module provides graph sparsification algorithms, including
DSpar and baseline methods.

Sparsification reduces the number of edges while attempting to
preserve important structural properties.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def sparsify_graph(
    G: nx.Graph,
    method: str = "dspar",
    ratio: float = 0.5,
    seed: Optional[int] = None,
    **kwargs,
) -> nx.Graph:
    """
    Sparsify a graph using the specified method.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    method : str, optional
        Sparsification method: 'dspar', 'random', 'degree_based'
        (default: 'dspar')
    ratio : float, optional
        Fraction of edges to retain (default: 0.5)
    seed : int, optional
        Random seed
    **kwargs
        Additional arguments for specific methods

    Returns
    -------
    nx.Graph
        Sparsified graph

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.complete_graph(10)
    >>> G_sparse = sparsify_graph(G, method='random', ratio=0.3)
    >>> G_sparse.number_of_edges() < G.number_of_edges()
    True
    """
    method = method.lower()

    if method == "dspar":
        return dspar_sparsification(G, ratio=ratio, seed=seed, **kwargs)
    elif method == "random":
        return random_sparsification(G, ratio=ratio, seed=seed)
    elif method == "degree_based":
        return degree_based_sparsification(G, ratio=ratio, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown sparsification method: {method}")


def compute_dspar_scores(G: nx.Graph) -> Dict[Tuple[int, int], float]:
    """
    Compute DSpar score for each edge.

    DSpar score: s(e) = 1/d_u + 1/d_v

    Higher score = edge connects low-degree nodes (more important to keep)
    Lower score = edge connects high-degree hubs (less important)

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    scores : dict
        Edge tuple (min, max) -> DSpar score
    """
    degrees = dict(G.degree())
    scores = {}

    for u, v in G.edges():
        score = 1.0 / degrees[u] + 1.0 / degrees[v]
        edge = (min(u, v), max(u, v))
        scores[edge] = score

    return scores


def dspar_sparsification(
    G: nx.Graph,
    ratio: float = 0.5,
    seed: Optional[int] = None,
    return_weights: bool = False,
) -> nx.Graph:
    """
    Sparsify using DSpar (Degree-based Sparsification).

    DSpar score: s(e) = 1/d_u + 1/d_v
    Higher score = edge connects low-degree nodes (more important to keep)
    Lower score = edge connects high-degree hubs (less important)

    Parameters
    ----------
    G : nx.Graph
        Input graph
    ratio : float, optional
        Fraction of edges to sample (default: 0.5)
    seed : int, optional
        Random seed
    return_weights : bool, optional
        If True, also return edge weights dictionary

    Returns
    -------
    nx.Graph
        Sparsified weighted graph
    weights : dict (only if return_weights=True)
        Edge weights after reweighting

    Notes
    -----
    Uses the original DSpar paper method:
    - Probabilistic sampling WITH replacement
    - Edge reweighting to preserve spectral properties: w'_e = k_e / (q * p_e)
      where k_e is the number of times edge e was sampled, q is total samples,
      and p_e is the sampling probability.
    """
    from collections import Counter

    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    if seed is not None:
        np.random.seed(seed)

    # Compute DSpar scores
    scores = compute_dspar_scores(G)
    edges = list(scores.keys())
    score_values = np.array([scores[e] for e in edges])

    m = len(edges)  # Original number of edges

    # Step 1: Compute sampling probabilities
    probs = score_values / score_values.sum()

    # Step 2: Number of samples to draw
    q = int(np.ceil(ratio * m))

    # Step 3: Sample WITH replacement
    sampled_indices = np.random.choice(
        len(edges),
        size=q,
        replace=True,  # WITH replacement
        p=probs
    )

    # Step 4: Count how many times each edge was sampled
    edge_counts = Counter(sampled_indices)

    # Step 5: Compute new weights
    # w'_e = k_e / (q * p_e)
    # This ensures unbiased estimation of original graph
    weights = {}
    for idx, count in edge_counts.items():
        edge = edges[idx]
        p_e = probs[idx]
        w_e = count / (q * p_e)
        weights[edge] = w_e

    # Step 6: Build weighted graph
    G_sparse = nx.Graph()
    G_sparse.add_nodes_from(G.nodes(data=True))

    for edge, weight in weights.items():
        u, v = edge
        G_sparse.add_edge(u, v, weight=weight)

    logger.debug(
        f"DSpar: {G.number_of_edges()} -> {G_sparse.number_of_edges()} edges"
    )

    if return_weights:
        return G_sparse, weights
    return G_sparse


def random_sparsification(
    G: nx.Graph,
    ratio: float = 0.5,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Randomly sparsify a graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    ratio : float, optional
        Fraction of edges to retain (default: 0.5)
    seed : int, optional
        Random seed

    Returns
    -------
    nx.Graph
        Sparsified graph
    """
    rng = np.random.default_rng(seed)

    G_sparse = G.copy()
    edges = list(G.edges())

    n_edges_target = max(1, int(len(edges) * ratio))
    n_to_remove = len(edges) - n_edges_target

    if n_to_remove <= 0:
        return G_sparse

    # Randomly select edges to remove
    edges_to_remove = rng.choice(len(edges), size=n_to_remove, replace=False)

    for idx in edges_to_remove:
        u, v = edges[idx]
        if G_sparse.has_edge(u, v):
            G_sparse.remove_edge(u, v)

    logger.debug(
        f"Random sparsification: {G.number_of_edges()} -> {G_sparse.number_of_edges()} edges"
    )

    return G_sparse


def degree_based_sparsification(
    G: nx.Graph,
    ratio: float = 0.5,
    seed: Optional[int] = None,
    strategy: str = "threshold",
) -> nx.Graph:
    """
    Sparsify based on node degrees.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    ratio : float, optional
        Fraction of edges to retain (default: 0.5)
    seed : int, optional
        Random seed
    strategy : str, optional
        Strategy for degree-based sparsification:
        - 'threshold': keep edges where both endpoints are below median degree
        - 'proportional': keep edges with probability proportional to 1/(d_u * d_v)
        - 'hub_removal': preferentially remove edges incident to high-degree nodes

    Returns
    -------
    nx.Graph
        Sparsified graph
    """
    rng = np.random.default_rng(seed)

    G_sparse = G.copy()
    edges = list(G.edges())
    degrees = dict(G.degree())

    n_edges_target = max(1, int(len(edges) * ratio))

    if strategy == "threshold":
        # Keep edges where both endpoints have degree below threshold
        median_degree = np.median(list(degrees.values()))

        # Score edges by max degree of endpoints
        edge_scores = [(u, v, max(degrees[u], degrees[v])) for u, v in edges]
        edge_scores.sort(key=lambda x: x[2])

        # Keep lowest-scored edges
        G_sparse.clear_edges()
        for u, v, _ in edge_scores[:n_edges_target]:
            G_sparse.add_edge(u, v)

    elif strategy == "proportional":
        # Keep edges with probability proportional to 1/(d_u * d_v)
        probs = np.array([1.0 / (degrees[u] * degrees[v]) for u, v in edges])
        probs = probs / probs.sum()

        # Sample edges to keep
        keep_indices = rng.choice(
            len(edges), size=n_edges_target, replace=False, p=probs
        )

        G_sparse.clear_edges()
        for idx in keep_indices:
            G_sparse.add_edge(*edges[idx])

    elif strategy == "hub_removal":
        # Remove edges incident to highest-degree nodes
        edge_scores = [(u, v, max(degrees[u], degrees[v])) for u, v in edges]
        edge_scores.sort(key=lambda x: -x[2])  # Highest first

        n_to_remove = len(edges) - n_edges_target
        for u, v, _ in edge_scores[:n_to_remove]:
            if G_sparse.has_edge(u, v):
                G_sparse.remove_edge(u, v)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    logger.debug(
        f"Degree-based sparsification ({strategy}): "
        f"{G.number_of_edges()} -> {G_sparse.number_of_edges()} edges"
    )

    return G_sparse


def spectral_sparsification(
    G: nx.Graph,
    ratio: float = 0.5,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Sparsify using spectral methods (edge importance based on Laplacian).

    This method attempts to preserve the spectral properties of the graph.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    ratio : float, optional
        Fraction of edges to retain (default: 0.5)
    seed : int, optional
        Random seed

    Returns
    -------
    nx.Graph
        Sparsified graph

    Notes
    -----
    This is a simplified spectral sparsification based on effective resistance.
    For large graphs, this can be computationally expensive.
    """
    rng = np.random.default_rng(seed)

    G_sparse = G.copy()
    edges = list(G.edges())

    n_edges_target = max(1, int(len(edges) * ratio))

    # Compute approximate effective resistance using random projections
    # This is a simplified approximation
    try:
        L = nx.laplacian_matrix(G).toarray()
        n = len(L)

        # Pseudo-inverse of Laplacian (regularized)
        L_reg = L + 0.01 * np.eye(n)
        L_pinv = np.linalg.pinv(L_reg)

        # Effective resistance for each edge
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        resistances = []
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
            resistances.append(r)

        resistances = np.array(resistances)

        # Sample edges with probability proportional to resistance
        probs = resistances / resistances.sum()

        keep_indices = rng.choice(
            len(edges), size=n_edges_target, replace=False, p=probs
        )

        G_sparse.clear_edges()
        for idx in keep_indices:
            G_sparse.add_edge(*edges[idx])

    except Exception as e:
        logger.warning(f"Spectral sparsification failed: {e}, falling back to random")
        return random_sparsification(G, ratio=ratio, seed=seed)

    return G_sparse


def analyze_sparsification_impact(
    G_original: nx.Graph,
    G_sparse: nx.Graph,
    communities: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Analyze the impact of sparsification on network properties.

    Parameters
    ----------
    G_original : nx.Graph
        Original graph
    G_sparse : nx.Graph
        Sparsified graph
    communities : Dict[int, int], optional
        Community assignments

    Returns
    -------
    Dict[str, Any]
        Analysis results including edge retention, connectivity, etc.
    """
    n_orig = G_original.number_of_edges()
    n_sparse = G_sparse.number_of_edges()

    analysis = {
        "original_edges": n_orig,
        "sparse_edges": n_sparse,
        "retention_ratio": n_sparse / n_orig if n_orig > 0 else 0,
        "original_connected": nx.is_connected(G_original),
        "sparse_connected": nx.is_connected(G_sparse),
    }

    # Component analysis
    if not nx.is_connected(G_sparse):
        components = list(nx.connected_components(G_sparse))
        analysis["n_components"] = len(components)
        analysis["largest_component_fraction"] = max(len(c) for c in components) / G_sparse.number_of_nodes()

    # Community-based analysis
    if communities is not None:
        from ..metrics.hub_bridging import classify_edges_by_hub_bridging

        inter_orig, intra_orig = classify_edges_by_hub_bridging(G_original, communities)
        inter_sparse, intra_sparse = classify_edges_by_hub_bridging(G_sparse, communities)

        analysis["inter_retention"] = len(inter_sparse) / max(len(inter_orig), 1)
        analysis["intra_retention"] = len(intra_sparse) / max(len(intra_orig), 1)
        analysis["inter_intra_ratio_change"] = (
            (len(inter_sparse) / max(len(intra_sparse), 1))
            / (len(inter_orig) / max(len(intra_orig), 1))
            if len(intra_orig) > 0 and len(intra_sparse) > 0
            else np.nan
        )

    return analysis
