"""
Hub-Bridging LFR Generator (Direct Generation)
==============================================

This module implements the Hub-Bridging LFR (HB-LFR) benchmark generator
using a direct generation approach. Instead of rewiring, edges are
placed directly according to a hub-bridging probability model.

This approach may be more efficient for extreme hub-bridging values
and provides different theoretical properties than the rewiring approach.

References
----------
.. [1] Your PhD thesis or publication on hub-bridging benchmarks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _generate_powerlaw_sequence(
    n: int,
    tau: float,
    min_val: int,
    max_val: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """
    Generate a power-law distributed sequence of integers.

    Parameters
    ----------
    n : int
        Number of values to generate
    tau : float
        Power-law exponent
    min_val : int
        Minimum value
    max_val : int
        Maximum value
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    NDArray[np.int64]
        Array of power-law distributed integers
    """
    # Use inverse transform sampling
    x = np.arange(min_val, max_val + 1, dtype=float)
    probs = x ** (-tau)
    probs /= probs.sum()

    return rng.choice(x.astype(int), size=n, p=probs)


def _generate_community_sizes(
    n: int,
    tau2: float,
    min_size: int,
    max_size: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Generate community sizes following a power-law distribution.

    Parameters
    ----------
    n : int
        Total number of nodes
    tau2 : float
        Power-law exponent for community sizes
    min_size : int
        Minimum community size
    max_size : int
        Maximum community size
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    List[int]
        List of community sizes summing to n
    """
    sizes = []
    remaining = n

    while remaining > 0:
        # Sample a size
        size = int(_generate_powerlaw_sequence(1, tau2, min_size, min(max_size, remaining), rng)[0])
        size = min(size, remaining)

        if size >= min_size:
            sizes.append(size)
            remaining -= size
        elif remaining <= max_size:
            # Handle remainder
            if sizes:
                sizes[-1] += remaining
            else:
                sizes.append(remaining)
            remaining = 0

    return sizes


def _compute_edge_probability(
    degree_u: int,
    degree_v: int,
    is_inter: bool,
    h: float,
    mu: float,
    normalization: float,
) -> float:
    """
    Compute edge probability based on hub-bridging model.

    For inter-community edges:
        P(u, v) ∝ (d_u * d_v)^(1 + h) * mu

    For intra-community edges:
        P(u, v) ∝ (d_u * d_v) * (1 - mu)

    Parameters
    ----------
    degree_u, degree_v : int
        Target degrees of endpoints
    is_inter : bool
        Whether this is an inter-community edge
    h : float
        Hub-bridging exponent
    mu : float
        Mixing parameter
    normalization : float
        Normalization constant

    Returns
    -------
    float
        Edge probability
    """
    base_prob = degree_u * degree_v

    if is_inter:
        prob = (base_prob ** (1 + h)) * mu
    else:
        prob = base_prob * (1 - mu)

    return prob / normalization


def hb_lfr_direct(
    n: int = 1000,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.3,
    h: float = 0.0,
    average_degree: float = 15.0,
    min_degree: int = 5,
    max_degree: Optional[int] = None,
    min_community: int = 20,
    max_community: Optional[int] = None,
    max_iterations: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate Hub-Bridging LFR graph using direct edge placement.

    This generator places edges according to a probability model that
    directly incorporates the hub-bridging exponent h.

    Parameters
    ----------
    n : int, optional
        Number of nodes (default: 1000)
    tau1 : float, optional
        Power-law exponent for degree distribution (default: 2.5)
    tau2 : float, optional
        Power-law exponent for community sizes (default: 1.5)
    mu : float, optional
        Mixing parameter (default: 0.3)
    h : float, optional
        Hub-bridging exponent (default: 0.0)
        - h = 0: Standard configuration model behavior
        - h > 0: Hubs prefer inter-community edges
    average_degree : float, optional
        Target average degree (default: 15.0)
    min_degree : int, optional
        Minimum node degree (default: 5)
    max_degree : int, optional
        Maximum node degree (default: n // 10)
    min_community : int, optional
        Minimum community size (default: 20)
    max_community : int, optional
        Maximum community size (default: n // 5)
    max_iterations : int, optional
        Maximum iterations for edge placement (default: 1000)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Tuple[nx.Graph, Dict[int, int]]
        (graph, communities) where communities maps node -> community_id

    Examples
    --------
    >>> G, communities = hb_lfr_direct(n=250, mu=0.3, h=0.5, seed=42)
    >>> G.number_of_nodes()
    250

    Notes
    -----
    This direct generation approach:
    1. Generates target degree sequence from power-law
    2. Assigns nodes to communities
    3. Places edges with probability proportional to degree product,
       modified by h for inter-community edges

    The advantage over rewiring is that the final graph directly
    reflects the hub-bridging model without depending on the
    rewiring dynamics.

    TODO: This is a stub implementation. Complete the edge placement
    algorithm to exactly match target degree sequence while respecting
    the hub-bridging probability model.
    """
    rng = np.random.default_rng(seed)

    if max_degree is None:
        max_degree = max(n // 10, int(average_degree * 3))
    if max_community is None:
        max_community = max(n // 5, min_community * 3)

    logger.info(
        f"Generating HB-LFR (direct): n={n}, tau1={tau1}, tau2={tau2}, "
        f"mu={mu}, h={h}, avg_deg={average_degree}"
    )

    # Step 1: Generate target degree sequence
    target_degrees = _generate_powerlaw_sequence(
        n, tau1, min_degree, max_degree, rng
    )

    # Adjust to achieve target average degree
    current_avg = target_degrees.mean()
    if current_avg > 0:
        scale = average_degree / current_avg
        target_degrees = np.clip(
            (target_degrees * scale).astype(int),
            min_degree,
            max_degree
        )

    # Ensure sum is even (for valid degree sequence)
    if target_degrees.sum() % 2 == 1:
        target_degrees[0] += 1

    # Step 2: Generate community structure
    community_sizes = _generate_community_sizes(
        n, tau2, min_community, max_community, rng
    )

    # Assign nodes to communities
    communities = {}
    node_idx = 0
    for comm_id, size in enumerate(community_sizes):
        for _ in range(size):
            if node_idx < n:
                communities[node_idx] = comm_id
                node_idx += 1

    # Handle any remaining nodes
    while node_idx < n:
        communities[node_idx] = len(community_sizes) - 1
        node_idx += 1

    # Step 3: Create graph and place edges
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Group nodes by community
    community_nodes: Dict[int, List[int]] = {}
    for node, comm in communities.items():
        if comm not in community_nodes:
            community_nodes[comm] = []
        community_nodes[comm].append(node)

    # TODO: Implement proper edge placement algorithm
    # For now, use a simplified approach based on configuration model
    # with hub-bridging modification

    # Create stub lists for each node based on degree
    stubs = []
    for node in range(n):
        stubs.extend([node] * target_degrees[node])

    rng.shuffle(stubs)

    # Pair up stubs to create edges
    edges_added = set()
    for i in range(0, len(stubs) - 1, 2):
        u, v = stubs[i], stubs[i + 1]

        # Skip self-loops and duplicates
        if u == v or (min(u, v), max(u, v)) in edges_added:
            continue

        # Determine if this should be inter or intra based on h and mu
        is_same_community = communities[u] == communities[v]

        # Apply hub-bridging probability modification
        degree_product = target_degrees[u] * target_degrees[v]

        if is_same_community:
            # For intra-community, accept with base probability
            accept_prob = 1.0 - mu
        else:
            # For inter-community, boost probability for high-degree pairs
            boost = (degree_product / (average_degree ** 2)) ** h
            accept_prob = min(1.0, mu * boost)

        if rng.random() < accept_prob or not is_same_community:
            edges_added.add((min(u, v), max(u, v)))
            G.add_edge(u, v)

    logger.info(
        f"Generated graph with {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges, "
        f"{len(community_sizes)} communities"
    )

    # Verify and report statistics
    actual_degrees = [d for _, d in G.degree()]
    logger.info(
        f"Degree stats: target_avg={average_degree:.1f}, "
        f"actual_avg={np.mean(actual_degrees):.1f}, "
        f"actual_max={max(actual_degrees)}"
    )

    return G, communities


def hb_lfr_direct_v2(
    n: int = 1000,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.3,
    h: float = 0.0,
    average_degree: float = 15.0,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Alternative direct generation using modified preferential attachment.

    This is an experimental alternative that grows the network using
    preferential attachment with hub-bridging modification.

    TODO: Implement this alternative approach.
    """
    raise NotImplementedError(
        "hb_lfr_direct_v2 is not yet implemented. "
        "Use hb_lfr_direct or hb_lfr (rewiring-based) instead."
    )
