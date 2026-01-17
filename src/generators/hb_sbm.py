"""
Hub-Bridging Stochastic Block Model Generator
=============================================

This module implements the Hub-Bridging SBM (HB-SBM) generator,
which extends the standard Stochastic Block Model with degree-dependent
edge probabilities to control hub-bridging behavior.

Unlike HB-LFR, which starts with power-law degrees, HB-SBM allows
for more controlled experiments with explicit probability matrices.

References
----------
.. [1] Your PhD thesis or publication on hub-bridging benchmarks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _assign_expected_degrees(
    n: int,
    community_sizes: List[int],
    degree_distribution: str = "uniform",
    mean_degree: float = 10.0,
    degree_heterogeneity: float = 0.5,
    rng: np.random.Generator = None,
) -> NDArray[np.float64]:
    """
    Assign expected degrees to nodes.

    Parameters
    ----------
    n : int
        Number of nodes
    community_sizes : List[int]
        Size of each community
    degree_distribution : str
        Type of degree distribution: 'uniform', 'exponential', 'powerlaw'
    mean_degree : float
        Target mean degree
    degree_heterogeneity : float
        Controls variance in degrees (0 = uniform, 1 = high variance)
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    NDArray[np.float64]
        Expected degree for each node
    """
    if rng is None:
        rng = np.random.default_rng()

    if degree_distribution == "uniform":
        # All nodes have same expected degree
        theta = np.ones(n) * mean_degree
    elif degree_distribution == "exponential":
        # Exponential distribution
        scale = mean_degree * degree_heterogeneity
        theta = rng.exponential(scale, size=n)
        theta = np.clip(theta, 1, mean_degree * 5)
        theta = theta * (mean_degree / theta.mean())  # Normalize to target mean
    elif degree_distribution == "powerlaw":
        # Power-law distribution
        alpha = 2.5 - degree_heterogeneity  # Lower alpha = higher variance
        alpha = max(2.01, alpha)  # Ensure finite mean
        theta = (rng.pareto(alpha - 1, size=n) + 1) * (mean_degree * (alpha - 2) / (alpha - 1))
        theta = np.clip(theta, 1, mean_degree * 10)
        theta = theta * (mean_degree / theta.mean())
    else:
        raise ValueError(f"Unknown degree distribution: {degree_distribution}")

    return theta


def _compute_hb_probability_matrix(
    theta: NDArray[np.float64],
    communities: Dict[int, int],
    p_in: float,
    p_out: float,
    h: float,
) -> NDArray[np.float64]:
    """
    Compute the hub-bridging probability matrix.

    For nodes u, v with expected degrees theta_u, theta_v:

    - Intra-community: P(u, v) = p_in * normalized_product
    - Inter-community: P(u, v) = p_out * (relative_degree_product)^h

    where relative_degree_product = (theta_u * theta_v) / mean(theta)^2

    When h > 0, high-degree pairs have increased inter-community edge probability,
    which increases the hub-bridging ratio.

    Parameters
    ----------
    theta : NDArray[np.float64]
        Expected degrees for each node
    communities : Dict[int, int]
        Community assignments
    p_in : float
        Base intra-community edge probability
    p_out : float
        Base inter-community edge probability
    h : float
        Hub-bridging exponent (0 = standard SBM, >0 = more hub-bridging)

    Returns
    -------
    NDArray[np.float64]
        n x n probability matrix
    """
    n = len(theta)

    # Compute relative degree products (mean = 1)
    mean_theta_sq = (theta.mean()) ** 2

    # Build probability matrix
    P = np.zeros((n, n))

    for u in range(n):
        for v in range(u + 1, n):
            same_community = communities[u] == communities[v]

            # Base degree product relative to mean
            relative_product = (theta[u] * theta[v]) / mean_theta_sq

            if same_community:
                # Intra-community: standard degree-corrected
                prob = p_in * relative_product
            else:
                # Inter-community: hub-bridging modification
                # h=0: same as intra (degree-corrected)
                # h>0: high-degree pairs boosted relative to intra
                prob = p_out * (relative_product ** (1 + h))

            # Ensure probability is valid
            prob = min(prob, 1.0)

            P[u, v] = prob
            P[v, u] = prob

    return P


def hb_sbm(
    n: int = 1000,
    k: int = 10,
    p_in: float = 0.3,
    p_out: float = 0.05,
    h: float = 0.0,
    community_sizes: Optional[List[int]] = None,
    degree_distribution: str = "uniform",
    mean_degree: float = 10.0,
    degree_heterogeneity: float = 0.5,
    theta_distribution: Optional[str] = None,
    degree_correction_scale: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a Hub-Bridging Stochastic Block Model graph.

    IMPROVED: Better degree correction parameterization for stronger
    hub-bridging control.

    This generator extends the standard SBM by making inter-community
    edge probabilities depend on node degrees, controlled by parameter h.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    k : int, optional
        Number of communities (default: 10)
    p_in : float, optional
        Base intra-community edge probability (default: 0.3)
    p_out : float, optional
        Base inter-community edge probability (default: 0.05)
    h : float, optional
        Hub-bridging exponent (default: 0.0)
        - h = 0: Standard degree-corrected SBM
        - h > 0: High-degree nodes more likely to form inter-community edges
    community_sizes : List[int], optional
        Sizes of each community. If None, communities are equal-sized.
    degree_distribution : str, optional
        (Legacy) Expected degree distribution: 'uniform', 'exponential', 'powerlaw'
        (default: 'uniform'). Use theta_distribution for improved control.
    mean_degree : float, optional
        Target mean degree (default: 10.0)
    degree_heterogeneity : float, optional
        Controls degree variance for non-uniform distributions (default: 0.5)
    theta_distribution : str, optional
        Distribution for degree corrections: 'exponential', 'power_law', 'lognormal'.
        If provided, overrides degree_distribution for improved hub-bridging control.
    degree_correction_scale : float, optional
        Scale parameter for degree heterogeneity when using theta_distribution.
        Higher values = more heterogeneous degrees (default: 1.0)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[nx.Graph, Dict[int, int]]
        (graph, communities) where communities maps node -> community_id

    Examples
    --------
    >>> G, communities = hb_sbm(n=100, k=5, h=0.5, seed=42)
    >>> G.number_of_nodes()
    100
    >>> len(set(communities.values()))
    5

    >>> # With power-law theta distribution for better hub-bridging control
    >>> G, communities = hb_sbm(
    ...     n=500, k=10, h=1.0,
    ...     theta_distribution='power_law',
    ...     degree_correction_scale=1.5,
    ...     seed=42
    ... )

    Notes
    -----
    The HB-SBM model generates graphs where:

    P(edge between u, v) = {
        p_in * relative_product,              if same community
        p_out * (relative_product)^(1+h),     if different communities
    }

    where relative_product = (theta_u * theta_v) / mean(theta)^2

    When h = 0, this reduces to the standard degree-corrected SBM.
    When h > 0, high-degree pairs (relative_product > 1) are boosted for
    inter-community edges, increasing the hub-bridging ratio.
    """
    rng = np.random.default_rng(seed)

    logger.info(
        f"Generating HB-SBM: n={n}, k={k}, p_in={p_in}, p_out={p_out}, "
        f"h={h}, theta_dist={theta_distribution or degree_distribution}"
    )

    # Determine community sizes
    if community_sizes is None:
        base_size = n // k
        remainder = n % k
        community_sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]

    if sum(community_sizes) != n:
        raise ValueError(f"Community sizes sum to {sum(community_sizes)}, expected {n}")

    # Assign nodes to communities (dict format)
    communities = {}
    node_idx = 0
    for comm_id, size in enumerate(community_sizes):
        for _ in range(size):
            communities[node_idx] = comm_id
            node_idx += 1

    # Generate theta (degree correction parameters)
    if theta_distribution is not None:
        # Use improved theta distribution
        theta = _generate_theta_improved(
            n=n,
            theta_distribution=theta_distribution,
            degree_correction_scale=degree_correction_scale,
            rng=rng,
        )
    else:
        # Use legacy degree distribution
        theta = _assign_expected_degrees(
            n=n,
            community_sizes=community_sizes,
            degree_distribution=degree_distribution,
            mean_degree=mean_degree,
            degree_heterogeneity=degree_heterogeneity,
            rng=rng,
        )

    # Compute probability matrix with improved formula
    P = _compute_hb_probability_matrix(
        theta=theta,
        communities=communities,
        p_in=p_in,
        p_out=p_out,
        h=h,
    )

    # Generate edges
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Sample edges according to probability matrix
    random_matrix = rng.random((n, n))
    edges = np.argwhere((random_matrix < P) & (np.triu(np.ones((n, n)), k=1) > 0))

    G.add_edges_from(edges.tolist())

    # Convert communities dict to list of sets for G.graph storage
    communities_list: List[set] = [set() for _ in range(k)]
    for node, comm_id in communities.items():
        communities_list[comm_id].add(node)

    # Store metadata in graph
    G.graph['communities'] = communities_list
    G.graph['n_communities'] = k
    G.graph['theta'] = theta
    G.graph['h'] = h
    G.graph['params'] = {
        'n': n, 'k': k, 'p_in': p_in, 'p_out': p_out, 'h': h,
        'theta_distribution': theta_distribution or degree_distribution,
        'degree_correction_scale': degree_correction_scale,
    }

    logger.info(
        f"Generated HB-SBM with {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    # Report statistics
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    try:
        rho_hb = compute_hub_bridging_ratio(G, communities)
        logger.info(f"Hub-bridging ratio: {rho_hb:.4f}")
    except Exception as e:
        logger.warning(f"Could not compute hub-bridging ratio: {e}")

    return G, communities


def _generate_theta_improved(
    n: int,
    theta_distribution: str,
    degree_correction_scale: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Generate improved theta (degree correction) parameters.

    Parameters
    ----------
    n : int
        Number of nodes
    theta_distribution : str
        Distribution type: 'exponential', 'power_law', 'lognormal'
    degree_correction_scale : float
        Scale parameter for heterogeneity
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    NDArray[np.float64]
        Theta values normalized to mean = 1
    """
    if theta_distribution == 'exponential':
        theta = rng.exponential(degree_correction_scale, n)
    elif theta_distribution == 'power_law':
        # Power-law distributed degrees (more realistic for social networks)
        # Pareto with alpha=2.5 gives power-law tail
        theta = (rng.pareto(2.5, n) + 1) * degree_correction_scale
    elif theta_distribution == 'lognormal':
        theta = rng.lognormal(0, degree_correction_scale, n)
    else:
        raise ValueError(f"Unknown theta_distribution: {theta_distribution}")

    # Normalize to mean = 1 for interpretability
    theta = theta / np.mean(theta)

    return theta


def hb_sbm_simple(
    n: int = 100,
    k: int = 4,
    p_in: float = 0.3,
    p_out: float = 0.05,
    h: float = 0.0,
    community_sizes: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a simplified Hub-Bridging SBM graph.

    This is a simplified version that directly implements the hub-bridging
    formula and stores communities in G.graph['communities'].

    Edge probability formula:
    - Intra-community: P(u,v) = theta_u * theta_v * p_in
    - Inter-community: P(u,v) = (theta_u * theta_v)^(1+h) * p_out

    Where theta_u is the normalized expected degree of node u.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 100)
    k : int, optional
        Number of communities (default: 4)
    p_in : float, optional
        Base intra-community edge probability (default: 0.3)
    p_out : float, optional
        Base inter-community edge probability (default: 0.05)
    h : float, optional
        Hub-bridging exponent (default: 0.0)
        - h = 0: Standard SBM behavior
        - h > 0: High-degree nodes more likely to form inter-community edges
        - h < 0: Low-degree nodes more likely to form inter-community edges
    community_sizes : List[int], optional
        Sizes of each community. If None, equal-sized communities.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Graph with G.graph['communities'] as list of sets

    Examples
    --------
    >>> G = hb_sbm_simple(n=100, k=4, h=0.5, seed=42)
    >>> G.number_of_nodes()
    100
    >>> 'communities' in G.graph
    True
    >>> len(G.graph['communities'])
    4
    """
    rng = np.random.default_rng(seed)

    # Determine community sizes
    if community_sizes is None:
        base_size = n // k
        remainder = n % k
        community_sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]

    if sum(community_sizes) != n:
        raise ValueError(f"Community sizes sum to {sum(community_sizes)}, expected {n}")

    # Assign nodes to communities
    communities_dict: Dict[int, int] = {}
    communities_list: List[set] = [set() for _ in range(k)]
    node_idx = 0
    for comm_id, size in enumerate(community_sizes):
        for _ in range(size):
            communities_dict[node_idx] = comm_id
            communities_list[comm_id].add(node_idx)
            node_idx += 1

    # Assign expected degrees (use power-law-like distribution)
    # Simple version: uniform + some variation
    theta = rng.uniform(0.5, 1.5, size=n)
    mean_theta_sq = (theta.mean()) ** 2

    # Build graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Generate edges
    for u in range(n):
        for v in range(u + 1, n):
            same_community = communities_dict[u] == communities_dict[v]

            # Base degree product relative to mean
            relative_product = (theta[u] * theta[v]) / mean_theta_sq

            if same_community:
                # Intra-community: standard degree-corrected
                prob = p_in * relative_product
            else:
                # Inter-community: hub-bridging modification
                # h=0: same as intra (degree-corrected)
                # h>0: high-degree pairs boosted relative to intra
                prob = p_out * (relative_product ** (1 + h))

            # Ensure valid probability
            prob = min(prob, 1.0)

            if rng.random() < prob:
                G.add_edge(u, v)

    # Store communities in graph
    G.graph['communities'] = communities_list
    G.graph['n_communities'] = k
    G.graph['h'] = h

    logger.info(
        f"Generated HB-SBM simple: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, h={h}"
    )

    return G


def hb_sbm_from_real_network(
    G_real: nx.Graph,
    communities_real: Dict[int, int],
    h: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate HB-SBM graph matching a real network's structure.

    This function estimates SBM parameters from a real network
    and generates a synthetic graph with controlled hub-bridging.

    Parameters
    ----------
    G_real : nx.Graph
        Real network to match
    communities_real : Dict[int, int]
        Community structure of real network
    h : float, optional
        Hub-bridging exponent (default: 0.0)
    seed : int, optional
        Random seed

    Returns
    -------
    Tuple[nx.Graph, Dict[int, int]]
        (synthetic graph, communities)

    Examples
    --------
    >>> import networkx as nx
    >>> G_real = nx.karate_club_graph()
    >>> communities_real = {n: 0 if n < 17 else 1 for n in G_real.nodes()}
    >>> G_synth, comm_synth = hb_sbm_from_real_network(
    ...     G_real, communities_real, h=0.5, seed=42
    ... )
    """
    n = G_real.number_of_nodes()

    # Count communities and their sizes
    comm_counts = {}
    for node, comm in communities_real.items():
        comm_counts[comm] = comm_counts.get(comm, 0) + 1

    k = len(comm_counts)
    community_sizes = [comm_counts[i] for i in sorted(comm_counts.keys())]

    # Estimate p_in and p_out
    intra_edges = 0
    inter_edges = 0
    possible_intra = 0
    possible_inter = 0

    nodes = list(G_real.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            same_comm = communities_real[u] == communities_real[v]
            has_edge = G_real.has_edge(u, v)

            if same_comm:
                possible_intra += 1
                if has_edge:
                    intra_edges += 1
            else:
                possible_inter += 1
                if has_edge:
                    inter_edges += 1

    p_in = intra_edges / max(possible_intra, 1)
    p_out = inter_edges / max(possible_inter, 1)

    # Estimate mean degree
    mean_degree = 2 * G_real.number_of_edges() / n

    # Estimate degree heterogeneity from coefficient of variation
    degrees = [d for _, d in G_real.degree()]
    cv = np.std(degrees) / max(np.mean(degrees), 1)
    degree_heterogeneity = min(cv / 2, 1.0)

    logger.info(
        f"Estimated parameters: p_in={p_in:.4f}, p_out={p_out:.4f}, "
        f"mean_deg={mean_degree:.2f}, heterogeneity={degree_heterogeneity:.2f}"
    )

    return hb_sbm(
        n=n,
        k=k,
        p_in=p_in,
        p_out=p_out,
        h=h,
        community_sizes=community_sizes,
        degree_distribution="exponential" if degree_heterogeneity > 0.3 else "uniform",
        mean_degree=mean_degree,
        degree_heterogeneity=degree_heterogeneity,
        seed=seed,
    )
