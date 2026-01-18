"""
Hub-Bridging LFR Generator (Rewiring-based)
===========================================

This module implements the Hub-Bridging LFR (HB-LFR) benchmark generator
using an edge rewiring approach. Starting from a standard LFR graph,
edges are rewired to achieve the target hub-bridging ratio controlled
by parameter h.

The rewiring process preserves:
- Degree distribution (exactly)
- Number of inter-community edges
- Community structure

References
----------
.. [1] Your PhD thesis or publication on hub-bridging benchmarks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Set

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def hb_lfr(
    n: int = 1000,
    tau1: float = 2.5,
    tau2: float = 1.5,
    mu: float = 0.3,
    h: float = 0.0,
    average_degree: Optional[float] = None,
    max_degree: Optional[int] = None,
    min_community: Optional[int] = None,
    max_community: Optional[int] = None,
    max_iters: int = 5000,
    tolerance: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, List[Set[int]]]:
    """
    Generate Hub-Bridged LFR benchmark graph.

    Algorithm:
    1. Generate standard LFR graph (h=0)
    2. If h>0, rewire inter-community edges to achieve target ρ_HB
    3. Preserve degree sequence exactly through edge swaps

    Parameters
    ----------
    n : int
        Number of nodes (default: 1000)
    tau1 : float
        Power-law exponent for degree distribution (default: 2.5)
    tau2 : float
        Power-law exponent for community size distribution (default: 1.5)
    mu : float
        Mixing parameter - fraction of inter-community edges per node (default: 0.3)
    h : float
        Hub-bridging parameter (default: 0.0)
        - h=0: Standard LFR (ρ_HB ≈ 1)
        - h>0: Hub-bridging (ρ_HB increases with h)
        - Recommended range: h ∈ [0, 2]
    average_degree : float
        Target average degree (default: 10.0)
    max_degree : int, optional
        Maximum node degree (default: n // 10)
    min_community : int, optional
        Minimum community size (default: n // 50)
    max_community : int, optional
        Maximum community size (default: n // 10)
    max_iters : int
        Maximum rewiring iterations (default: 5000)
    tolerance : float
        Convergence tolerance for ρ_HB (default: 0.05)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    G : networkx.Graph
        Generated graph with G.graph['communities'] attribute
    communities : list of sets
        Community assignments

    Examples
    --------
    >>> G, communities = hb_lfr(n=500, h=1.0, seed=42)
    >>> from src.metrics.hub_bridging import compute_hub_bridging_ratio
    >>> rho = compute_hub_bridging_ratio(G, communities)
    >>> print(f"Hub-bridging ratio: {rho:.2f}")

    Notes
    -----
    Rewiring preserves:
    - Exact degree sequence
    - Number of inter-community edges
    - Community assignments

    Does NOT preserve:
    - Specific edge identities
    - Local clustering (may change slightly)
    """
    rng = np.random.default_rng(seed)

    # Set defaults based on n
    if average_degree is None:
        average_degree = 10.0  # NetworkX LFR default
    if max_degree is None:
        max_degree = max(n // 10, 20)
    if min_community is None:
        min_community = max(n // 50, 10)
    if max_community is None:
        max_community = max(n // 10, 50)

    # Ensure constraints are satisfiable
    min_community = min(min_community, n // 2)
    max_community = min(max_community, n)
    max_community = max(max_community, min_community + 1)

    logger.info(f"Generating HB-LFR: n={n}, h={h:.2f}, mu={mu:.2f}")

    # Step 1: Generate base LFR
    try:
        G = nx.generators.community.LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=average_degree,
            max_degree=max_degree,
            min_community=min_community,
            max_community=max_community,
            seed=seed,
        )
    except nx.ExceededMaxIterations as e:
        raise ValueError(f"LFR generation failed (exceeded max iterations): {e}")
    except Exception as e:
        raise ValueError(f"LFR generation failed: {e}")

    # Extract communities from node attributes
    communities_dict: Dict[int, int] = {}
    comm_to_nodes: Dict[int, Set[int]] = {}

    for node in G.nodes():
        # LFR stores communities as a frozenset in node attributes
        comm_set = G.nodes[node].get('community', frozenset())
        # Take first community if node belongs to multiple (shouldn't happen in LFR)
        comm_id = min(comm_set) if comm_set else 0

        if comm_id not in comm_to_nodes:
            comm_to_nodes[comm_id] = set()
        comm_to_nodes[comm_id].add(node)
        communities_dict[node] = comm_id

    # Convert to list of sets format
    communities = [comm_to_nodes[i] for i in sorted(comm_to_nodes.keys())]

    # Clean up node attributes
    for node in G.nodes():
        if 'community' in G.nodes[node]:
            del G.nodes[node]['community']

    # Store in graph
    G.graph['communities'] = communities
    G.graph['n_communities'] = len(communities)
    G.graph['params'] = {
        'n': n, 'tau1': tau1, 'tau2': tau2, 'mu': mu, 'h': h,
        'average_degree': average_degree,
    }

    # If h=0, return standard LFR
    if h == 0.0:
        logger.info("h=0, returning standard LFR")
        return G, communities

    # Step 2: Compute target ρ_HB
    # Empirical calibration: ρ ≈ 1 + 1.5*(1 - exp(-0.8*h))
    # This captures the saturation behavior observed in HB-SBM
    rho_target = 1.0 + 1.5 * (1.0 - np.exp(-0.8 * h))

    logger.info(f"Target ρ_HB = {rho_target:.3f}")

    # Step 3: Rewire to achieve target
    G = _rewire_for_hub_bridging(
        G=G,
        communities_dict=communities_dict,
        rho_target=rho_target,
        h=h,
        max_iters=max_iters,
        tolerance=tolerance,
        rng=rng,
    )

    return G, communities


def _rewire_for_hub_bridging(
    G: nx.Graph,
    communities_dict: Dict[int, int],
    rho_target: float,
    h: float,
    max_iters: int,
    tolerance: float,
    rng: np.random.Generator,
) -> nx.Graph:
    """
    Rewire graph to achieve target hub-bridging ratio.

    Strategy:
    - Remove low-degree inter-community edges
    - Add high-degree inter-community edges
    - Preserve total number of inter-community edges
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    degrees = dict(G.degree())
    nodes = list(G.nodes())
    n = len(nodes)

    # Compute initial ρ_HB
    communities_list = []
    comm_ids = sorted(set(communities_dict.values()))
    for comm_id in comm_ids:
        communities_list.append({n for n, c in communities_dict.items() if c == comm_id})

    rho_current = compute_hub_bridging_ratio(G, communities_dict)
    logger.info(f"Initial ρ_HB = {rho_current:.3f}, target = {rho_target:.3f}")

    # If already at target, return
    if abs(rho_current - rho_target) < tolerance:
        logger.info("Already at target, no rewiring needed")
        return G

    # Track progress
    successful_rewires = 0
    stall_count = 0
    best_rho = rho_current
    best_diff = abs(rho_current - rho_target)
    no_improvement_count = 0  # Track iterations without improvement
    last_check_diff = best_diff

    for iteration in range(max_iters):
        # Perform rewiring step
        if rho_current < rho_target:
            success = _rewire_step_increase(G, communities_dict, degrees, h, rng)
        else:
            success = _rewire_step_decrease(G, communities_dict, degrees, h, rng)

        if success:
            successful_rewires += 1
            stall_count = 0

            # Recompute ρ_HB periodically (expensive)
            if successful_rewires % 50 == 0:
                rho_current = compute_hub_bridging_ratio(G, communities_dict)
                diff = abs(rho_current - rho_target)

                if diff < best_diff:
                    best_diff = diff
                    best_rho = rho_current
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Check convergence
                if diff < tolerance:
                    logger.info(
                        f"Converged at iteration {iteration}: ρ={rho_current:.3f} "
                        f"(target={rho_target:.3f})"
                    )
                    break

                # EARLY TERMINATION: No improvement for 5 consecutive checks
                if no_improvement_count >= 5:
                    logger.warning(
                        f"Early stop at iteration {iteration}: no improvement "
                        f"(best ρ={best_rho:.3f}, target={rho_target:.3f})"
                    )
                    break

                # EARLY TERMINATION: Target clearly unreachable (diff > 0.3 after 1000 iters)
                if iteration > 1000 and diff > 0.3 and diff >= last_check_diff * 0.98:
                    logger.warning(
                        f"Early stop at iteration {iteration}: target unreachable "
                        f"(ρ={rho_current:.3f}, target={rho_target:.3f})"
                    )
                    break

                last_check_diff = diff

                # Log progress (less frequently)
                if successful_rewires % 1000 == 0:
                    logger.info(
                        f"Iteration {iteration}: ρ={rho_current:.3f} "
                        f"(target={rho_target:.3f}, diff={diff:.4f})"
                    )
        else:
            stall_count += 1

            # Check for stagnation (reduced from 200 to 100)
            if stall_count > 100:
                rho_current = compute_hub_bridging_ratio(G, communities_dict)
                logger.warning(
                    f"Rewiring stalled at iteration {iteration}: ρ={rho_current:.3f}"
                )
                break

    else:
        rho_current = compute_hub_bridging_ratio(G, communities_dict)
        logger.warning(
            f"Max iterations reached. Final ρ={rho_current:.3f} "
            f"(target={rho_target:.3f})"
        )

    logger.info(f"Rewiring complete: {successful_rewires} successful swaps")
    return G


def _rewire_step_increase(
    G: nx.Graph,
    communities_dict: Dict[int, int],
    degrees: Dict[int, int],
    h: float,
    rng: np.random.Generator,
) -> bool:
    """
    Perform one rewiring step to increase ρ_HB.

    Strategy:
    1. Remove a low-degree-product inter-community edge
    2. Add a high-degree-product inter-community edge
    3. This increases mean(d_u * d_v | inter) → increases ρ_HB

    Returns True if successful, False if no valid rewiring found.
    """
    # Get inter-community edges
    inter_edges = [
        (u, v) for u, v in G.edges()
        if communities_dict[u] != communities_dict[v]
    ]

    if len(inter_edges) < 2:
        return False

    # Select low-degree inter-edge to remove
    # Probability ∝ 1/(d_u * d_v)
    edge_products = np.array([
        degrees[u] * degrees[v] for u, v in inter_edges
    ], dtype=np.float64)

    # Avoid numerical issues
    edge_products = np.clip(edge_products, 1.0, None)

    weights_remove = 1.0 / edge_products
    weights_remove = weights_remove / weights_remove.sum()

    idx_remove = rng.choice(len(inter_edges), p=weights_remove)
    u_remove, v_remove = inter_edges[idx_remove]

    # Find candidate high-degree non-edges to add
    # Sample candidate pairs (don't check all O(n²) pairs)
    nodes = list(G.nodes())
    n_candidates = min(500, len(nodes) * 5)

    candidates = []
    cand_weights = []

    for _ in range(n_candidates):
        u = rng.choice(nodes)
        v = rng.choice(nodes)

        if u == v:
            continue
        if communities_dict[u] == communities_dict[v]:
            continue
        if G.has_edge(u, v):
            continue

        # Valid candidate - different communities, not already connected
        prod = degrees[u] * degrees[v]
        candidates.append((u, v))
        cand_weights.append(prod ** max(h, 1.0))  # Boost high-degree pairs

    if len(candidates) == 0:
        return False

    # Select high-degree pair to add
    cand_weights = np.array(cand_weights, dtype=np.float64)
    cand_weights = cand_weights / cand_weights.sum()

    idx_add = rng.choice(len(candidates), p=cand_weights)
    u_add, v_add = candidates[idx_add]

    # Check that we're actually improving (new product > old product)
    old_product = degrees[u_remove] * degrees[v_remove]
    new_product = degrees[u_add] * degrees[v_add]

    if new_product <= old_product:
        # Not an improvement, reject this swap
        return False

    # Perform swap
    G.remove_edge(u_remove, v_remove)
    G.add_edge(u_add, v_add)

    return True


def _rewire_step_decrease(
    G: nx.Graph,
    communities_dict: Dict[int, int],
    degrees: Dict[int, int],
    h: float,
    rng: np.random.Generator,
) -> bool:
    """
    Perform one rewiring step to decrease ρ_HB.

    Inverse of increase: remove high-degree, add low-degree.
    (Rarely needed, but included for completeness)

    Returns True if successful, False if no valid rewiring found.
    """
    # Get inter-community edges
    inter_edges = [
        (u, v) for u, v in G.edges()
        if communities_dict[u] != communities_dict[v]
    ]

    if len(inter_edges) < 2:
        return False

    # Select HIGH-degree inter-edge to remove
    edge_products = np.array([
        degrees[u] * degrees[v] for u, v in inter_edges
    ], dtype=np.float64)

    weights_remove = edge_products ** max(h, 1.0)
    weights_remove = weights_remove / weights_remove.sum()

    idx_remove = rng.choice(len(inter_edges), p=weights_remove)
    u_remove, v_remove = inter_edges[idx_remove]

    # Find LOW-degree pairs to add
    nodes = list(G.nodes())
    n_candidates = min(500, len(nodes) * 5)

    candidates = []
    cand_weights = []

    for _ in range(n_candidates):
        u = rng.choice(nodes)
        v = rng.choice(nodes)

        if u == v:
            continue
        if communities_dict[u] == communities_dict[v]:
            continue
        if G.has_edge(u, v):
            continue

        prod = degrees[u] * degrees[v]
        candidates.append((u, v))
        cand_weights.append(1.0 / max(prod, 1.0))  # Prefer low-degree pairs

    if len(candidates) == 0:
        return False

    cand_weights = np.array(cand_weights, dtype=np.float64)
    cand_weights = cand_weights / cand_weights.sum()

    idx_add = rng.choice(len(candidates), p=cand_weights)
    u_add, v_add = candidates[idx_add]

    # Check that we're actually improving (new product < old product)
    old_product = degrees[u_remove] * degrees[v_remove]
    new_product = degrees[u_add] * degrees[v_add]

    if new_product >= old_product:
        return False

    # Perform swap
    G.remove_edge(u_remove, v_remove)
    G.add_edge(u_add, v_add)

    return True


def hb_lfr_rewiring(
    G: nx.Graph,
    communities: List[Set[int]],
    h: float,
    max_iters: int = 5000,
    tolerance: float = 0.05,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Apply hub-bridging rewiring to an existing graph.

    This is useful for converting any graph with known communities
    into a hub-bridging version.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    communities : list of sets
        Community assignments
    h : float
        Hub-bridging parameter
    max_iters : int
        Maximum rewiring iterations
    tolerance : float
        Convergence tolerance
    seed : int, optional
        Random seed

    Returns
    -------
    nx.Graph
        Rewired graph (modified in place)
    """
    rng = np.random.default_rng(seed)

    # Create node->community mapping
    communities_dict = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            communities_dict[node] = comm_id

    # Compute target
    rho_target = 1.0 + 1.5 * (1.0 - np.exp(-0.8 * h))

    return _rewire_for_hub_bridging(
        G=G,
        communities_dict=communities_dict,
        rho_target=rho_target,
        h=h,
        max_iters=max_iters,
        tolerance=tolerance,
        rng=rng,
    )
