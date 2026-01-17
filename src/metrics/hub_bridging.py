"""
Hub-Bridging Metrics
====================

This module provides metrics for computing hub-bridging properties
of networks, including the hub-bridging ratio (rho_HB) and DSpar
separation metrics.

The hub-bridging ratio quantifies the tendency for high-degree nodes
(hubs) to form inter-community edges versus intra-community edges.

Key Metrics
-----------
- rho_HB: Hub-bridging ratio = E[d_u*d_v | inter] / E[d_u*d_v | intra]
- delta: DSpar separation = E[1/d_u + 1/d_v | intra] - E[1/d_u + 1/d_v | inter]

These metrics are inversely related: high rho_HB implies high delta.

References
----------
.. [1] Your PhD thesis or relevant publication
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Type alias for communities
Communities = Union[List[Set[int]], List[List[int]], Dict[int, int]]


def _parse_communities(communities: Communities) -> Dict[int, int]:
    """
    Parse communities into standard dict format.

    Handles multiple input formats and converts them to a unified
    {node_id: community_id} dictionary representation.

    Parameters
    ----------
    communities : list of sets, list of lists, or dict
        Community assignments in various formats:
        - List of sets: [{node1, node2}, {node3, node4}]
        - List of lists: [[node1, node2], [node3, node4]]
        - Dict: {node_id: community_id}

    Returns
    -------
    Dict[int, int]
        Mapping from node_id to community_id

    Examples
    --------
    >>> _parse_communities([{0, 1}, {2, 3}])
    {0: 0, 1: 0, 2: 1, 3: 1}

    >>> _parse_communities({0: 'A', 1: 'A', 2: 'B'})
    {0: 'A', 1: 'A', 2: 'B'}

    >>> _parse_communities([[0, 1, 2], [3, 4, 5]])
    {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    """
    if isinstance(communities, dict):
        logger.debug(f"Communities already in dict format with {len(communities)} nodes")
        return communities

    # List of sets or list of lists
    node_to_community: Dict[int, int] = {}
    for comm_id, members in enumerate(communities):
        for node in members:
            node_to_community[node] = comm_id

    logger.debug(
        f"Parsed {len(communities)} communities with {len(node_to_community)} total nodes"
    )
    return node_to_community


def _partition_edges(
    G: nx.Graph,
    node_to_community: Dict[int, int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Partition edges into intra-community and inter-community.

    Parameters
    ----------
    G : networkx.Graph
        The input graph
    node_to_community : dict
        Mapping from node_id to community_id

    Returns
    -------
    E_intra : list of tuples
        List of (u, v) tuples for intra-community edges
    E_inter : list of tuples
        List of (u, v) tuples for inter-community edges

    Examples
    --------
    >>> G = nx.Graph([(0, 1), (1, 2), (2, 3)])
    >>> node_to_comm = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> intra, inter = _partition_edges(G, node_to_comm)
    >>> len(intra), len(inter)
    (2, 1)
    """
    E_intra: List[Tuple[int, int]] = []
    E_inter: List[Tuple[int, int]] = []

    for u, v in G.edges():
        comm_u = node_to_community.get(u)
        comm_v = node_to_community.get(v)

        if comm_u is None or comm_v is None:
            logger.warning(f"Node {u} or {v} not in communities, skipping edge")
            continue

        if comm_u == comm_v:
            E_intra.append((u, v))
        else:
            E_inter.append((u, v))

    logger.debug(f"Partitioned edges: {len(E_intra)} intra, {len(E_inter)} inter")
    return E_intra, E_inter


def compute_dspar_score(u: int, v: int, degrees: Dict[int, int]) -> float:
    """
    Compute DSpar score for edge (u, v).

    The DSpar score is the sum of inverse degrees of the endpoints.
    Higher scores indicate edges between low-degree nodes.

    Parameters
    ----------
    u : int
        First node ID
    v : int
        Second node ID
    degrees : dict
        Degree dictionary {node_id: degree}

    Returns
    -------
    float
        DSpar score = 1/d_u + 1/d_v

    Examples
    --------
    >>> degrees = {0: 10, 1: 5, 2: 2}
    >>> compute_dspar_score(0, 1, degrees)
    0.3
    >>> compute_dspar_score(1, 2, degrees)
    0.7
    """
    return 1.0 / degrees[u] + 1.0 / degrees[v]


def compute_hub_bridging_ratio(
    G: nx.Graph,
    communities: Communities,
) -> float:
    """
    Compute hub-bridging ratio rho_HB.

    The hub-bridging ratio quantifies whether inter-community edges
    preferentially connect high-degree nodes compared to intra-community edges.

    Definition:
        rho_HB = E[d_u * d_v | inter-edge] / E[d_u * d_v | intra-edge]

    Where:
        - d_u, d_v are degrees of endpoints
        - Inter-edge: edge connecting different communities
        - Intra-edge: edge within same community

    Parameters
    ----------
    G : networkx.Graph
        The input graph
    communities : list of sets, list of lists, or dict
        Community assignments. Either:
        - List of sets: [{node1, node2}, {node3, node4}]
        - List of lists: [[node1, node2], [node3, node4]]
        - Dict: {node_id: community_id}

    Returns
    -------
    float
        Hub-bridging ratio.
        - rho_HB = 1: no hub-bridging (degree-independent mixing)
        - rho_HB > 1: hub-bridging (inter-edges prefer hubs)
        - rho_HB < 1: hub-isolation (inter-edges avoid hubs)

    Raises
    ------
    ValueError
        If graph has no inter-community or no intra-community edges

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> communities = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21},
    ...               {9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}]
    >>> rho = compute_hub_bridging_ratio(G, communities)
    >>> print(f"Hub-bridging ratio: {rho:.3f}")

    Notes
    -----
    If either E_inter or E_intra is empty, raises ValueError.
    In practice, all reasonable community structures have both types.

    The hub-bridging ratio is the key metric for validating that
    HB-LFR and HB-SBM generators correctly control the relationship
    between node degree and edge placement.
    """
    # Parse communities into standard format
    node_to_community = _parse_communities(communities)

    # Validate that all nodes are covered
    missing_nodes = set(G.nodes()) - set(node_to_community.keys())
    if missing_nodes:
        logger.warning(
            f"{len(missing_nodes)} nodes not in communities: {list(missing_nodes)[:5]}..."
        )

    # Get degrees
    degrees = dict(G.degree())

    # Partition edges
    E_intra, E_inter = _partition_edges(G, node_to_community)

    # Check for empty partitions
    if len(E_inter) == 0:
        raise ValueError(
            "Graph has no inter-community edges. "
            "Check that communities span multiple groups connected by edges."
        )

    if len(E_intra) == 0:
        raise ValueError(
            "Graph has no intra-community edges. "
            "Check that communities have internal edges."
        )

    # Compute degree products for inter-community edges
    inter_products = np.array([degrees[u] * degrees[v] for u, v in E_inter])
    mean_inter = np.mean(inter_products)

    # Compute degree products for intra-community edges
    intra_products = np.array([degrees[u] * degrees[v] for u, v in E_intra])
    mean_intra = np.mean(intra_products)

    # Compute ratio
    if mean_intra == 0:
        logger.warning("Mean intra-community degree product is 0, returning inf")
        return np.inf

    rho_hb = mean_inter / mean_intra

    logger.debug(
        f"rho_HB = {rho_hb:.4f} | "
        f"inter: n={len(E_inter)}, mean_prod={mean_inter:.2f} | "
        f"intra: n={len(E_intra)}, mean_prod={mean_intra:.2f}"
    )

    return float(rho_hb)


def compute_dspar_separation(
    G: nx.Graph,
    communities: Communities,
) -> float:
    """
    Compute DSpar separation delta.

    DSpar separation measures the difference in average DSpar scores
    between intra-community and inter-community edges.

    Definition:
        delta = mu_intra - mu_inter

    Where:
        mu_intra = E[1/d_u + 1/d_v | intra-edge]
        mu_inter = E[1/d_u + 1/d_v | inter-edge]

    DSpar score for edge (u,v): s(u,v) = 1/d_u + 1/d_v

    Parameters
    ----------
    G : networkx.Graph
        The input graph
    communities : list of sets, list of lists, or dict
        Community assignments

    Returns
    -------
    float
        DSpar separation.
        - delta > 0: intra-edges have higher DSpar (lower degree products)
        - delta < 0: inter-edges have higher DSpar
        - delta ~ 0: no systematic difference

    Raises
    ------
    ValueError
        If graph has no inter-community or no intra-community edges

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> communities = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21},
    ...               {9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33}]
    >>> delta = compute_dspar_separation(G, communities)
    >>> print(f"DSpar separation: {delta:.4f}")

    Notes
    -----
    DSpar separation is related to hub-bridging ratio but uses harmonic
    mean structure instead of arithmetic mean of degree products.
    They are inversely correlated:
    - If rho_HB > 1 (hub-bridging), then delta > 0
    - If rho_HB < 1 (hub-isolation), then delta < 0

    This relationship follows from Proposition 1 in the paper.
    """
    # Parse communities into standard format
    node_to_community = _parse_communities(communities)

    # Get degrees
    degrees = dict(G.degree())

    # Partition edges
    E_intra, E_inter = _partition_edges(G, node_to_community)

    # Check for empty partitions
    if len(E_inter) == 0:
        raise ValueError(
            "Graph has no inter-community edges. "
            "Check that communities span multiple groups connected by edges."
        )

    if len(E_intra) == 0:
        raise ValueError(
            "Graph has no intra-community edges. "
            "Check that communities have internal edges."
        )

    # Compute DSpar scores for inter-community edges
    inter_scores = np.array([compute_dspar_score(u, v, degrees) for u, v in E_inter])
    mu_inter = np.mean(inter_scores)

    # Compute DSpar scores for intra-community edges
    intra_scores = np.array([compute_dspar_score(u, v, degrees) for u, v in E_intra])
    mu_intra = np.mean(intra_scores)

    # Compute separation
    delta = mu_intra - mu_inter

    logger.debug(
        f"delta = {delta:.4f} | "
        f"mu_intra = {mu_intra:.4f}, mu_inter = {mu_inter:.4f}"
    )

    return float(delta)


def classify_edges_by_hub_bridging(
    G: nx.Graph,
    communities: Communities,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Classify edges as inter-community (bridging) or intra-community.

    This is a convenience wrapper around _partition_edges that handles
    community parsing.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : list of sets, list of lists, or dict
        Community assignments

    Returns
    -------
    Tuple[List, List]
        (inter_edges, intra_edges) - lists of edge tuples

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> communities = {n: 0 if n < 17 else 1 for n in G.nodes()}
    >>> inter, intra = classify_edges_by_hub_bridging(G, communities)
    >>> len(inter) + len(intra) == G.number_of_edges()
    True
    """
    node_to_community = _parse_communities(communities)
    E_intra, E_inter = _partition_edges(G, node_to_community)
    return E_inter, E_intra


def compute_hub_bridging_profile(
    G: nx.Graph,
    communities: Communities,
    degree_percentiles: Optional[List[float]] = None,
) -> Dict[str, NDArray[np.float64]]:
    """
    Compute hub-bridging ratio across different degree ranges.

    This function computes rho_HB for edges where at least one endpoint
    has degree in different percentile ranges, providing a more detailed
    view of hub-bridging behavior.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : list of sets, list of lists, or dict
        Community assignments
    degree_percentiles : List[float], optional
        Percentile thresholds for degree ranges.
        Default: [0, 25, 50, 75, 90, 95, 99, 100]

    Returns
    -------
    Dict[str, NDArray]
        Dictionary with keys:
        - 'percentiles': the percentile thresholds used
        - 'degree_thresholds': actual degree values at each percentile
        - 'rho_hb_cumulative': rho_HB for edges with max(deg_u, deg_v) >= threshold
        - 'n_edges': number of edges in each category

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.barabasi_albert_graph(100, 3, seed=42)
    >>> communities = [{n for n in G.nodes() if n % 2 == 0},
    ...               {n for n in G.nodes() if n % 2 == 1}]
    >>> profile = compute_hub_bridging_profile(G, communities)
    >>> len(profile['percentiles'])
    8
    """
    if degree_percentiles is None:
        degree_percentiles = [0, 25, 50, 75, 90, 95, 99, 100]

    node_to_community = _parse_communities(communities)
    degrees = dict(G.degree())
    degree_values = np.array(list(degrees.values()))

    # Compute degree thresholds
    thresholds = np.percentile(degree_values, degree_percentiles)

    # Partition edges
    E_intra, E_inter = _partition_edges(G, node_to_community)

    # Compute rho_HB for edges where max degree >= threshold
    rho_values = []
    n_edges_list = []

    for threshold in thresholds:
        # Filter edges
        inter_filtered = [
            (u, v) for u, v in E_inter
            if max(degrees[u], degrees[v]) >= threshold
        ]
        intra_filtered = [
            (u, v) for u, v in E_intra
            if max(degrees[u], degrees[v]) >= threshold
        ]

        n_edges = len(inter_filtered) + len(intra_filtered)
        n_edges_list.append(n_edges)

        if len(inter_filtered) == 0 or len(intra_filtered) == 0:
            rho_values.append(np.nan)
            continue

        inter_products = [degrees[u] * degrees[v] for u, v in inter_filtered]
        intra_products = [degrees[u] * degrees[v] for u, v in intra_filtered]

        mean_inter = np.mean(inter_products)
        mean_intra = np.mean(intra_products)

        if mean_intra > 0:
            rho_values.append(mean_inter / mean_intra)
        else:
            rho_values.append(np.nan)

    return {
        "percentiles": np.array(degree_percentiles),
        "degree_thresholds": thresholds,
        "rho_hb_cumulative": np.array(rho_values),
        "n_edges": np.array(n_edges_list),
    }
