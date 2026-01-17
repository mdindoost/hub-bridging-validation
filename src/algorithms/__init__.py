"""
Algorithms Module
=================

This module provides algorithm implementations for community
detection and graph sparsification used in validation experiments.

Submodules
----------
community_detection
    Community detection algorithms and evaluation metrics
sparsification
    Graph sparsification methods including DSpar
"""

from .community_detection import (
    detect_communities,
    detect_communities_louvain,
    detect_communities_leiden,
    detect_communities_label_propagation,
    compute_nmi,
    compute_ari,
)
from .sparsification import (
    sparsify_graph,
    dspar_sparsification,
    compute_dspar_scores,
    random_sparsification,
    degree_based_sparsification,
)

__all__ = [
    # Community detection
    "detect_communities",
    "detect_communities_louvain",
    "detect_communities_leiden",
    "detect_communities_label_propagation",
    "compute_nmi",
    "compute_ari",
    # Sparsification
    "sparsify_graph",
    "dspar_sparsification",
    "compute_dspar_scores",
    "random_sparsification",
    "degree_based_sparsification",
]
