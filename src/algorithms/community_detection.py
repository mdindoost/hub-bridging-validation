"""
Community Detection Module
==========================

This module provides community detection algorithms and evaluation
metrics for comparing detected communities to ground truth.

Supported algorithms:
- Louvain
- Leiden
- Label Propagation
- Infomap (if available)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

logger = logging.getLogger(__name__)


def detect_communities(
    G: nx.Graph,
    algorithm: str = "louvain",
    **kwargs,
) -> Dict[int, int]:
    """
    Detect communities using the specified algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    algorithm : str, optional
        Algorithm to use: 'louvain', 'leiden', 'label_propagation', 'infomap'
        (default: 'louvain')
    **kwargs
        Additional arguments passed to the algorithm

    Returns
    -------
    Dict[int, int]
        Dictionary mapping node -> community_id

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> communities = detect_communities(G, algorithm='louvain')
    >>> len(set(communities.values())) > 1
    True
    """
    algorithm = algorithm.lower()

    if algorithm == "louvain":
        return detect_communities_louvain(G, **kwargs)
    elif algorithm == "leiden":
        return detect_communities_leiden(G, **kwargs)
    elif algorithm == "label_propagation":
        return detect_communities_label_propagation(G, **kwargs)
    elif algorithm == "infomap":
        return detect_communities_infomap(G, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def detect_communities_louvain(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """
    Detect communities using the Louvain algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    resolution : float, optional
        Resolution parameter (default: 1.0)
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[int, int]
        Community assignments
    """
    try:
        import community as community_louvain
    except ImportError:
        logger.warning("python-louvain not installed, using NetworkX implementation")
        # Fall back to NetworkX implementation
        communities_gen = nx.community.louvain_communities(
            G, resolution=resolution, seed=seed
        )
        communities = {}
        for comm_id, nodes in enumerate(communities_gen):
            for node in nodes:
                communities[node] = comm_id
        return communities

    # Use python-louvain
    partition = community_louvain.best_partition(
        G, resolution=resolution, random_state=seed
    )
    return partition


def detect_communities_leiden(
    G: nx.Graph,
    resolution: float = 1.0,
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """
    Detect communities using the Leiden algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    resolution : float, optional
        Resolution parameter (default: 1.0)
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[int, int]
        Community assignments
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        logger.warning("leidenalg or igraph not installed, falling back to Louvain")
        return detect_communities_louvain(G, resolution=resolution, seed=seed)

    # Convert NetworkX graph to igraph
    edges = list(G.edges())
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    ig_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in edges]
    ig_graph = ig.Graph(n=len(nodes), edges=ig_edges)

    # Run Leiden
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )

    # Convert back to node -> community dict
    communities = {}
    for comm_id, members in enumerate(partition):
        for idx in members:
            communities[nodes[idx]] = comm_id

    return communities


def detect_communities_label_propagation(
    G: nx.Graph,
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """
    Detect communities using label propagation.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[int, int]
        Community assignments
    """
    if seed is not None:
        np.random.seed(seed)

    communities_gen = nx.community.label_propagation_communities(G)

    communities = {}
    for comm_id, nodes in enumerate(communities_gen):
        for node in nodes:
            communities[node] = comm_id

    return communities


def detect_communities_infomap(
    G: nx.Graph,
    seed: Optional[int] = None,
) -> Dict[int, int]:
    """
    Detect communities using Infomap algorithm.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[int, int]
        Community assignments
    """
    try:
        from infomap import Infomap
    except ImportError:
        logger.warning("infomap not installed, falling back to Louvain")
        return detect_communities_louvain(G, seed=seed)

    # Create Infomap object
    im = Infomap(silent=True, seed=seed if seed is not None else 0)

    # Build network
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    for u, v in G.edges():
        im.add_link(node_to_idx[u], node_to_idx[v])

    # Run algorithm
    im.run()

    # Extract communities
    communities = {}
    for node in im.tree:
        if node.is_leaf:
            communities[nodes[node.node_id]] = node.module_id

    return communities


def compute_nmi(
    true_communities: Dict[int, int],
    detected_communities: Dict[int, int],
    nodes: Optional[List[int]] = None,
) -> float:
    """
    Compute Normalized Mutual Information between two partitions.

    Parameters
    ----------
    true_communities : Dict[int, int]
        Ground truth community assignments
    detected_communities : Dict[int, int]
        Detected community assignments
    nodes : List[int], optional
        Nodes to consider. If None, uses intersection of both dicts.

    Returns
    -------
    float
        NMI score (0 = no mutual information, 1 = perfect match)

    Examples
    --------
    >>> true = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> detected = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> compute_nmi(true, detected)
    1.0
    """
    if nodes is None:
        nodes = list(set(true_communities.keys()) & set(detected_communities.keys()))

    if len(nodes) == 0:
        return 0.0

    true_labels = [true_communities.get(n, -1) for n in nodes]
    detected_labels = [detected_communities.get(n, -1) for n in nodes]

    return float(normalized_mutual_info_score(true_labels, detected_labels))


def compute_ari(
    true_communities: Dict[int, int],
    detected_communities: Dict[int, int],
    nodes: Optional[List[int]] = None,
) -> float:
    """
    Compute Adjusted Rand Index between two partitions.

    Parameters
    ----------
    true_communities : Dict[int, int]
        Ground truth community assignments
    detected_communities : Dict[int, int]
        Detected community assignments
    nodes : List[int], optional
        Nodes to consider

    Returns
    -------
    float
        ARI score (-1 to 1, with 1 = perfect match, 0 = random)

    Examples
    --------
    >>> true = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> detected = {0: 0, 1: 0, 2: 1, 3: 1}
    >>> compute_ari(true, detected)
    1.0
    """
    if nodes is None:
        nodes = list(set(true_communities.keys()) & set(detected_communities.keys()))

    if len(nodes) == 0:
        return 0.0

    true_labels = [true_communities.get(n, -1) for n in nodes]
    detected_labels = [detected_communities.get(n, -1) for n in nodes]

    return float(adjusted_rand_score(true_labels, detected_labels))


def evaluate_detection(
    G: nx.Graph,
    true_communities: Dict[int, int],
    detected_communities: Dict[int, int],
) -> Dict[str, float]:
    """
    Comprehensive evaluation of community detection results.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    true_communities : Dict[int, int]
        Ground truth
    detected_communities : Dict[int, int]
        Detected partition

    Returns
    -------
    Dict[str, float]
        Evaluation metrics including NMI, ARI, and modularity
    """
    nodes = list(G.nodes())

    # Compute clustering metrics
    nmi = compute_nmi(true_communities, detected_communities, nodes)
    ari = compute_ari(true_communities, detected_communities, nodes)

    # Compute modularity of detected partition
    community_sets = {}
    for node, comm in detected_communities.items():
        if comm not in community_sets:
            community_sets[comm] = set()
        community_sets[comm].add(node)

    Q_detected = nx.community.modularity(G, list(community_sets.values()))

    # Compute modularity of true partition
    true_sets = {}
    for node, comm in true_communities.items():
        if comm not in true_sets:
            true_sets[comm] = set()
        true_sets[comm].add(node)

    Q_true = nx.community.modularity(G, list(true_sets.values()))

    return {
        "nmi": nmi,
        "ari": ari,
        "modularity_detected": Q_detected,
        "modularity_true": Q_true,
        "n_communities_detected": len(community_sets),
        "n_communities_true": len(true_sets),
    }
