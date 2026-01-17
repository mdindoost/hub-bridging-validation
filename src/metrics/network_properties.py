"""
Network Properties Module
=========================

This module provides comprehensive network property measurements
for comparing synthetic and real networks.

Properties include degree distribution statistics, clustering,
path lengths, rich-club coefficients, and community-related metrics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import stats
from collections import Counter

logger = logging.getLogger(__name__)


def compute_degree_distribution_stats(
    G: nx.Graph,
    fit_powerlaw: bool = True,
) -> Dict[str, Any]:
    """
    Compute comprehensive degree distribution statistics.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    fit_powerlaw : bool, optional
        If True, fit a power-law distribution (default: True)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'degrees': array of all degrees
        - 'mean': mean degree
        - 'std': standard deviation
        - 'min': minimum degree
        - 'max': maximum degree
        - 'median': median degree
        - 'skewness': skewness of distribution
        - 'kurtosis': kurtosis of distribution
        - 'gini': Gini coefficient (inequality measure)
        - 'powerlaw_alpha': power-law exponent (if fit_powerlaw=True)
        - 'powerlaw_xmin': minimum x for power-law fit

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barabasi_albert_graph(100, 3, seed=42)
    >>> stats = compute_degree_distribution_stats(G)
    >>> stats['mean'] > 0
    True
    """
    degrees = np.array([d for _, d in G.degree()])

    result = {
        "degrees": degrees,
        "mean": float(np.mean(degrees)),
        "std": float(np.std(degrees)),
        "min": int(np.min(degrees)),
        "max": int(np.max(degrees)),
        "median": float(np.median(degrees)),
        "skewness": float(stats.skew(degrees)),
        "kurtosis": float(stats.kurtosis(degrees)),
        "gini": _compute_gini(degrees),
    }

    if fit_powerlaw:
        try:
            import powerlaw
            fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
            result["powerlaw_alpha"] = float(fit.power_law.alpha)
            result["powerlaw_xmin"] = float(fit.power_law.xmin)
            result["powerlaw_sigma"] = float(fit.power_law.sigma)
        except Exception as e:
            logger.warning(f"Power-law fitting failed: {e}")
            result["powerlaw_alpha"] = np.nan
            result["powerlaw_xmin"] = np.nan
            result["powerlaw_sigma"] = np.nan

    return result


def _compute_gini(values: NDArray[np.float64]) -> float:
    """
    Compute Gini coefficient for measuring inequality.

    Parameters
    ----------
    values : NDArray
        Array of values

    Returns
    -------
    float
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative = np.cumsum(sorted_values)
    return float(
        (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumulative[-1]) - (n + 1) / n
    )


def compute_rich_club_coefficient(
    G: nx.Graph,
    k: Optional[int] = None,
    normalized: bool = True,
) -> Union[float, Dict[int, float]]:
    """
    Compute the rich-club coefficient.

    The rich-club coefficient measures the tendency of high-degree
    nodes to be interconnected.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    k : int, optional
        Degree threshold. If None, returns coefficients for all k.
    normalized : bool, optional
        If True, normalize by random graph expectation (default: True)

    Returns
    -------
    Union[float, Dict[int, float]]
        Rich-club coefficient for given k, or dict of all coefficients

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barabasi_albert_graph(100, 3, seed=42)
    >>> rc = compute_rich_club_coefficient(G, k=10)
    >>> rc >= 0
    True
    """
    if G.number_of_edges() == 0:
        return {} if k is None else 0.0

    # Compute raw rich-club coefficients
    rc = nx.rich_club_coefficient(G, normalized=False)

    if normalized:
        # Generate random graph with same degree sequence for normalization
        try:
            # Use configuration model for random baseline
            degree_sequence = [d for _, d in G.degree()]
            G_random = nx.configuration_model(degree_sequence)
            G_random = nx.Graph(G_random)  # Remove parallel edges
            G_random.remove_edges_from(nx.selfloop_edges(G_random))

            rc_random = nx.rich_club_coefficient(G_random, normalized=False)

            # Normalize
            rc_normalized = {}
            for degree in rc:
                if degree in rc_random and rc_random[degree] > 0:
                    rc_normalized[degree] = rc[degree] / rc_random[degree]
                else:
                    rc_normalized[degree] = rc[degree]
            rc = rc_normalized
        except Exception as e:
            logger.warning(f"Rich-club normalization failed: {e}")

    if k is not None:
        return rc.get(k, 0.0)

    return rc


def compute_participation_coefficient(
    G: nx.Graph,
    communities: Dict[int, int],
) -> NDArray[np.float64]:
    """
    Compute participation coefficient for all nodes.

    The participation coefficient measures how uniformly distributed
    a node's edges are across communities. High values indicate
    nodes that connect to many communities.

    P_i = 1 - sum_c (k_ic / k_i)^2

    where k_ic is the number of edges node i has to community c,
    and k_i is the total degree of node i.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : Dict[int, int]
        Dictionary mapping node -> community_id

    Returns
    -------
    NDArray[np.float64]
        Array of participation coefficients for each node

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
    >>> P = compute_participation_coefficient(G, communities)
    >>> len(P) == G.number_of_nodes()
    True
    >>> all(0 <= p <= 1 for p in P)
    True
    """
    nodes = list(G.nodes())
    P = np.zeros(len(nodes))

    for i, node in enumerate(nodes):
        neighbors = list(G.neighbors(node))
        k_i = len(neighbors)

        if k_i == 0:
            P[i] = 0.0
            continue

        # Count edges to each community
        community_counts = Counter(communities.get(n, -1) for n in neighbors)

        # Compute participation coefficient
        sum_squared = sum((count / k_i) ** 2 for count in community_counts.values())
        P[i] = 1.0 - sum_squared

    return P


def compute_within_module_degree(
    G: nx.Graph,
    communities: Dict[int, int],
) -> NDArray[np.float64]:
    """
    Compute within-module degree z-score for all nodes.

    The within-module degree z-score measures how well-connected
    a node is within its own community compared to other nodes
    in the same community.

    z_i = (k_i^within - mean(k^within_c)) / std(k^within_c)

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : Dict[int, int]
        Dictionary mapping node -> community_id

    Returns
    -------
    NDArray[np.float64]
        Array of within-module degree z-scores

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
    >>> z = compute_within_module_degree(G, communities)
    >>> len(z) == G.number_of_nodes()
    True
    """
    nodes = list(G.nodes())
    z = np.zeros(len(nodes))

    # Group nodes by community
    community_nodes: Dict[int, List[int]] = {}
    for node, comm in communities.items():
        if comm not in community_nodes:
            community_nodes[comm] = []
        community_nodes[comm].append(node)

    # Compute within-community degrees
    within_degrees: Dict[int, int] = {}
    for node in nodes:
        node_comm = communities.get(node, -1)
        neighbors = set(G.neighbors(node))
        same_comm_neighbors = [n for n in neighbors if communities.get(n, -1) == node_comm]
        within_degrees[node] = len(same_comm_neighbors)

    # Compute z-scores within each community
    for comm, comm_nodes in community_nodes.items():
        degrees = [within_degrees[n] for n in comm_nodes]
        mean_deg = np.mean(degrees)
        std_deg = np.std(degrees)

        for node in comm_nodes:
            node_idx = nodes.index(node)
            if std_deg > 0:
                z[node_idx] = (within_degrees[node] - mean_deg) / std_deg
            else:
                z[node_idx] = 0.0

    return z


def compute_clustering_stats(
    G: nx.Graph,
) -> Dict[str, float]:
    """
    Compute clustering coefficient statistics.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - 'global': global clustering coefficient
        - 'average': average local clustering coefficient
        - 'transitivity': transitivity (ratio of triangles)
    """
    return {
        "global": float(nx.transitivity(G)),
        "average": float(nx.average_clustering(G)),
        "transitivity": float(nx.transitivity(G)),
    }


def compute_path_length_stats(
    G: nx.Graph,
    sample_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute path length statistics.

    Parameters
    ----------
    G : nx.Graph
        Input graph (should be connected for meaningful results)
    sample_size : int, optional
        If provided, sample this many node pairs for efficiency

    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - 'average_path_length': average shortest path length
        - 'diameter': graph diameter (longest shortest path)
        - 'radius': graph radius (minimum eccentricity)
    """
    if not nx.is_connected(G):
        # Use largest connected component
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        logger.warning("Graph not connected, using largest component")

    if G.number_of_nodes() < 2:
        return {"average_path_length": 0.0, "diameter": 0, "radius": 0}

    try:
        if sample_size is not None and G.number_of_nodes() > sample_size:
            # Sample-based estimation
            nodes = list(G.nodes())
            sampled = np.random.choice(nodes, size=min(sample_size, len(nodes)), replace=False)
            lengths = []
            for source in sampled:
                path_lengths = nx.single_source_shortest_path_length(G, source)
                lengths.extend(path_lengths.values())
            avg_path = np.mean(lengths)
            # Approximate diameter
            diameter = max(lengths)
            radius = min(max(nx.single_source_shortest_path_length(G, n).values())
                        for n in sampled[:min(10, len(sampled))])
        else:
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            radius = nx.radius(G)

        return {
            "average_path_length": float(avg_path),
            "diameter": int(diameter),
            "radius": int(radius),
        }
    except Exception as e:
        logger.error(f"Path length computation failed: {e}")
        return {"average_path_length": np.nan, "diameter": np.nan, "radius": np.nan}


def compute_modularity(
    G: nx.Graph,
    communities: Dict[int, int],
) -> float:
    """
    Compute modularity of a partition.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : Dict[int, int]
        Dictionary mapping node -> community_id

    Returns
    -------
    float
        Modularity value
    """
    # Convert to list of sets format
    community_sets: Dict[int, set] = {}
    for node, comm in communities.items():
        if comm not in community_sets:
            community_sets[comm] = set()
        community_sets[comm].add(node)

    partition = list(community_sets.values())
    return float(nx.community.modularity(G, partition))


def comprehensive_network_properties(
    G: nx.Graph,
    communities: Optional[Dict[int, int]] = None,
    compute_expensive: bool = True,
) -> Dict[str, Any]:
    """
    Compute comprehensive network properties.

    This function computes a wide range of network properties useful
    for comparing synthetic and real networks.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : Dict[int, int], optional
        Community assignments. If None, community-related metrics
        are skipped.
    compute_expensive : bool, optional
        If True, compute expensive metrics like path lengths
        (default: True)

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all computed properties organized by category

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
    >>> props = comprehensive_network_properties(G, communities)
    >>> 'basic' in props
    True
    >>> 'degree' in props
    True
    """
    logger.info(f"Computing properties for graph with {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges")

    properties: Dict[str, Any] = {}

    # Basic properties
    properties["basic"] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "is_connected": nx.is_connected(G),
        "n_components": nx.number_connected_components(G),
    }

    # Degree distribution
    properties["degree"] = compute_degree_distribution_stats(G)

    # Clustering
    properties["clustering"] = compute_clustering_stats(G)

    # Path lengths (can be expensive)
    if compute_expensive:
        sample_size = 1000 if G.number_of_nodes() > 1000 else None
        properties["path_length"] = compute_path_length_stats(G, sample_size=sample_size)

    # Rich-club (sample of values)
    if compute_expensive:
        rc = compute_rich_club_coefficient(G, normalized=True)
        if isinstance(rc, dict):
            k_values = sorted(rc.keys())
            if k_values:
                properties["rich_club"] = {
                    "coefficients": {k: rc[k] for k in k_values[:20]},  # First 20
                    "max_k": max(k_values),
                }
            else:
                properties["rich_club"] = {"coefficients": {}, "max_k": 0}
        else:
            properties["rich_club"] = {"coefficients": {}, "max_k": 0}

    # Community-related metrics
    if communities is not None:
        properties["community"] = {
            "n_communities": len(set(communities.values())),
            "modularity": compute_modularity(G, communities),
        }

        # Participation coefficient
        P = compute_participation_coefficient(G, communities)
        properties["community"]["participation"] = {
            "mean": float(np.mean(P)),
            "std": float(np.std(P)),
            "max": float(np.max(P)),
        }

        # Within-module degree
        z = compute_within_module_degree(G, communities)
        properties["community"]["within_module_degree"] = {
            "mean": float(np.mean(z)),
            "std": float(np.std(z)),
        }

        # Hub-bridging ratio (import locally to avoid circular import)
        from .hub_bridging import compute_hub_bridging_ratio, compute_dspar_separation

        try:
            properties["hub_bridging"] = {
                "rho_hb": compute_hub_bridging_ratio(G, communities),
                "delta_dspar": compute_dspar_separation(G, communities),
            }
        except Exception as e:
            logger.warning(f"Hub-bridging metrics failed: {e}")
            properties["hub_bridging"] = {"rho_hb": np.nan, "delta_dspar": np.nan}

    return properties
