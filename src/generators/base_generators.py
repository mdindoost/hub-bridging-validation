"""
Base Generators Module
======================

This module provides wrappers for standard benchmark generators
(LFR, SBM, Planted Partition) used as baselines for comparison.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def generate_lfr(
    n: int = 1000,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.3,
    average_degree: Optional[float] = None,
    min_degree: Optional[int] = None,
    max_degree: Optional[int] = None,
    min_community: Optional[int] = None,
    max_community: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a standard LFR benchmark graph.

    Wraps NetworkX's LFR benchmark generator with convenient defaults
    and returns both the graph and community assignments.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    tau1 : float, optional
        Power-law exponent for degree distribution (default: 2.5)
    tau2 : float, optional
        Power-law exponent for community size distribution (default: 1.5)
    mu : float, optional
        Mixing parameter - fraction of edges that are inter-community
        (default: 0.3)
    average_degree : float, optional
        Target average degree. If None, uses min_degree * 1.5
    min_degree : int, optional
        Minimum node degree (default: computed from average_degree)
    max_degree : int, optional
        Maximum node degree (default: n // 10)
    min_community : int, optional
        Minimum community size (default: n // 50)
    max_community : int, optional
        Maximum community size (default: n // 10)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[nx.Graph, Dict[int, int]]
        (graph, communities) where communities maps node -> community_id

    Examples
    --------
    >>> G, communities = generate_lfr(n=250, mu=0.3, seed=42)
    >>> G.number_of_nodes()
    250
    >>> len(set(communities.values())) > 1
    True

    Notes
    -----
    The LFR (Lancichinetti-Fortunato-Radicchi) benchmark generates
    networks with power-law degree and community size distributions,
    making it suitable for testing community detection algorithms.
    """
    # Set defaults based on n
    if max_degree is None:
        max_degree = max(n // 10, 20)
    if min_community is None:
        min_community = max(n // 50, 10)
    if max_community is None:
        max_community = max(n // 10, 50)

    # Handle degree specification - NetworkX requires exactly ONE of min_degree or average_degree
    # We prefer average_degree if provided, otherwise use min_degree
    use_average_degree = average_degree is not None

    if use_average_degree:
        # Use average_degree, don't pass min_degree
        pass
    else:
        # Use min_degree
        if min_degree is None:
            min_degree = 10
        # Don't set average_degree

    # Ensure constraints are satisfiable
    if min_degree is not None:
        min_degree = min(min_degree, max_degree - 1)
    min_community = min(min_community, n // 2)
    max_community = min(max_community, n)
    max_community = max(max_community, min_community + 1)

    logger.info(
        f"Generating LFR: n={n}, tau1={tau1}, tau2={tau2}, mu={mu}, "
        f"avg_deg={average_degree}, min_deg={min_degree}, max_deg={max_degree}"
    )

    try:
        # Build kwargs - only include one of average_degree or min_degree
        lfr_kwargs = {
            "n": n,
            "tau1": tau1,
            "tau2": tau2,
            "mu": mu,
            "max_degree": max_degree,
            "min_community": min_community,
            "max_community": max_community,
            "seed": seed,
        }

        if use_average_degree:
            lfr_kwargs["average_degree"] = average_degree
        else:
            lfr_kwargs["min_degree"] = min_degree

        G = nx.LFR_benchmark_graph(**lfr_kwargs)

        # Extract community assignments
        communities = {}
        for node in G.nodes():
            # LFR stores communities as a set in node attributes
            comm_set = G.nodes[node].get("community", {0})
            # Take first community if node belongs to multiple
            communities[node] = min(comm_set)

        # Remove community attribute from nodes (clean up)
        for node in G.nodes():
            if "community" in G.nodes[node]:
                del G.nodes[node]["community"]

        logger.info(
            f"Generated LFR with {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, "
            f"{len(set(communities.values()))} communities"
        )

        return G, communities

    except nx.ExceededMaxIterations as e:
        logger.error(f"LFR generation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LFR generation: {e}")
        raise


def generate_sbm(
    n: int = 1000,
    k: int = 10,
    p_in: float = 0.3,
    p_out: float = 0.05,
    community_sizes: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a Stochastic Block Model graph.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    k : int, optional
        Number of communities (default: 10)
    p_in : float, optional
        Intra-community edge probability (default: 0.3)
    p_out : float, optional
        Inter-community edge probability (default: 0.05)
    community_sizes : List[int], optional
        Sizes of each community. If None, communities are equal-sized.
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[nx.Graph, Dict[int, int]]
        (graph, communities) where communities maps node -> community_id

    Examples
    --------
    >>> G, communities = generate_sbm(n=100, k=5, p_in=0.4, p_out=0.1, seed=42)
    >>> G.number_of_nodes()
    100
    >>> len(set(communities.values()))
    5

    Notes
    -----
    The SBM generates graphs where edge probability depends only on
    community membership. It produces graphs with uniform degree
    distributions within communities.
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine community sizes
    if community_sizes is None:
        # Equal-sized communities
        base_size = n // k
        remainder = n % k
        community_sizes = [base_size + (1 if i < remainder else 0) for i in range(k)]

    if sum(community_sizes) != n:
        raise ValueError(f"Community sizes sum to {sum(community_sizes)}, expected {n}")

    logger.info(
        f"Generating SBM: n={n}, k={k}, p_in={p_in}, p_out={p_out}, "
        f"sizes={community_sizes}"
    )

    # Build probability matrix
    p_matrix = np.full((k, k), p_out)
    np.fill_diagonal(p_matrix, p_in)

    # Generate graph using NetworkX
    G = nx.stochastic_block_model(
        sizes=community_sizes,
        p=p_matrix.tolist(),
        seed=seed,
    )

    # Extract community assignments from block attribute
    communities = {}
    for node in G.nodes():
        communities[node] = G.nodes[node].get("block", 0)

    # Clean up node attributes
    for node in G.nodes():
        if "block" in G.nodes[node]:
            del G.nodes[node]["block"]

    logger.info(
        f"Generated SBM with {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )

    return G, communities


def generate_planted_partition(
    n: int = 1000,
    k: int = 10,
    p_in: float = 0.3,
    p_out: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a Planted Partition Model graph.

    This is a special case of SBM with equal-sized communities.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    k : int, optional
        Number of communities (default: 10)
    p_in : float, optional
        Intra-community edge probability (default: 0.3)
    p_out : float, optional
        Inter-community edge probability (default: 0.05)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[nx.Graph, Dict[int, int]]
        (graph, communities) where communities maps node -> community_id

    Examples
    --------
    >>> G, communities = generate_planted_partition(n=100, k=4, seed=42)
    >>> all(sum(1 for c in communities.values() if c == i) == 25 for i in range(4))
    True
    """
    return generate_sbm(n=n, k=k, p_in=p_in, p_out=p_out, community_sizes=None, seed=seed)


def generate_configuration_model(
    degree_sequence: List[int],
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a random graph with a given degree sequence.

    Parameters
    ----------
    degree_sequence : List[int]
        Target degree for each node
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Random graph with approximately the given degree sequence

    Notes
    -----
    The configuration model may produce graphs with self-loops and
    parallel edges, which are removed. The final degrees may differ
    slightly from the target.
    """
    if sum(degree_sequence) % 2 != 0:
        # Make sum even by adding 1 to a random degree
        degree_sequence = list(degree_sequence)
        degree_sequence[0] += 1

    G = nx.configuration_model(degree_sequence, seed=seed)
    G = nx.Graph(G)  # Remove parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))  # Remove self-loops

    return G


def generate_powerlaw_cluster_graph(
    n: int = 1000,
    m: int = 3,
    p: float = 0.5,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a power-law cluster graph (Holme-Kim model).

    This model generates scale-free networks with tunable clustering.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    m : int, optional
        Number of edges to add for each new node (default: 3)
    p : float, optional
        Probability of adding a triangle after adding an edge (default: 0.5)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Power-law cluster graph
    """
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def generate_standard_lfr(
    n: int = 1000,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.3,
    average_degree: Optional[float] = None,
    min_degree: Optional[int] = None,
    max_degree: Optional[int] = None,
    min_community: Optional[int] = None,
    max_community: Optional[int] = None,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Generate a standard LFR benchmark graph with communities stored in G.graph.

    This is a simplified wrapper that returns a single graph object with
    communities stored as G.graph['communities'] (list of sets format).

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    tau1 : float, optional
        Power-law exponent for degree distribution (default: 2.5)
    tau2 : float, optional
        Power-law exponent for community size distribution (default: 1.5)
    mu : float, optional
        Mixing parameter - fraction of edges that are inter-community
        (default: 0.3)
    average_degree : float, optional
        Target average degree
    min_degree : int, optional
        Minimum node degree (default: 10)
    max_degree : int, optional
        Maximum node degree (default: n // 10)
    min_community : int, optional
        Minimum community size (default: n // 50)
    max_community : int, optional
        Maximum community size (default: n // 10)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    nx.Graph
        Graph with G.graph['communities'] as list of sets

    Examples
    --------
    >>> G = generate_standard_lfr(n=250, mu=0.3, seed=42)
    >>> G.number_of_nodes()
    250
    >>> 'communities' in G.graph
    True
    >>> isinstance(G.graph['communities'], list)
    True
    >>> all(isinstance(c, set) for c in G.graph['communities'])
    True
    """
    # Generate using the tuple-returning version
    G, communities_dict = generate_lfr(
        n=n,
        tau1=tau1,
        tau2=tau2,
        mu=mu,
        average_degree=average_degree,
        min_degree=min_degree,
        max_degree=max_degree,
        min_community=min_community,
        max_community=max_community,
        seed=seed,
    )

    # Convert dict format to list of sets
    comm_to_nodes: Dict[int, set] = {}
    for node, comm_id in communities_dict.items():
        if comm_id not in comm_to_nodes:
            comm_to_nodes[comm_id] = set()
        comm_to_nodes[comm_id].add(node)

    # Sort by community ID and convert to list
    communities_list = [comm_to_nodes[i] for i in sorted(comm_to_nodes.keys())]

    # Store in graph
    G.graph['communities'] = communities_list
    G.graph['n_communities'] = len(communities_list)

    return G


def extract_lfr_params(G: nx.Graph) -> Dict[str, Any]:
    """
    Extract LFR-like parameters from a real network.

    This function estimates parameters that could be used to generate
    a synthetic LFR network with similar properties.

    Parameters
    ----------
    G : nx.Graph
        Input network

    Returns
    -------
    Dict[str, Any]
        Dictionary with estimated parameters:
        - n: number of nodes
        - tau1: estimated power-law exponent for degree distribution
        - mu: estimated mixing parameter (requires communities)
        - average_degree: mean degree
        - min_degree: minimum degree
        - max_degree: maximum degree
        - degree_sequence: actual degree sequence

    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> params = extract_lfr_params(G)
    >>> params['n']
    34
    >>> 'tau1' in params
    True
    """
    degrees = np.array([d for _, d in G.degree()])

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Basic stats
    params = {
        'n': n,
        'm': m,
        'average_degree': 2 * m / n if n > 0 else 0,
        'min_degree': int(degrees.min()) if len(degrees) > 0 else 0,
        'max_degree': int(degrees.max()) if len(degrees) > 0 else 0,
        'degree_sequence': degrees.tolist(),
    }

    # Estimate power-law exponent tau1
    # Using MLE estimator: tau1 = 1 + n / sum(ln(d_i / d_min))
    if len(degrees) > 0 and params['min_degree'] > 0:
        d_min = params['min_degree']
        log_sum = np.sum(np.log(degrees / d_min))
        if log_sum > 0:
            params['tau1'] = 1.0 + n / log_sum
        else:
            params['tau1'] = 2.5  # Default
    else:
        params['tau1'] = 2.5

    # Estimate mixing parameter if communities are available
    if 'communities' in G.graph:
        communities = G.graph['communities']
        # Convert to dict format
        node_to_comm = {}
        for comm_id, members in enumerate(communities):
            for node in members:
                node_to_comm[node] = comm_id

        inter_edges = 0
        for u, v in G.edges():
            if node_to_comm.get(u, -1) != node_to_comm.get(v, -2):
                inter_edges += 1

        params['mu'] = inter_edges / m if m > 0 else 0
    else:
        params['mu'] = None  # Cannot estimate without communities

    # Estimate community size parameters
    if 'communities' in G.graph:
        comm_sizes = [len(c) for c in G.graph['communities']]
        params['min_community'] = min(comm_sizes)
        params['max_community'] = max(comm_sizes)
        params['n_communities'] = len(comm_sizes)

        # Estimate tau2 using similar MLE approach
        if min(comm_sizes) > 0:
            s_min = min(comm_sizes)
            log_sum = np.sum(np.log(np.array(comm_sizes) / s_min))
            if log_sum > 0:
                params['tau2'] = 1.0 + len(comm_sizes) / log_sum
            else:
                params['tau2'] = 1.5
        else:
            params['tau2'] = 1.5
    else:
        params['min_community'] = None
        params['max_community'] = None
        params['tau2'] = None

    return params
