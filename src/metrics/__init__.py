"""
Metrics Module
==============

This module provides metrics for analyzing hub-bridging properties
and network characteristics.

Submodules
----------
hub_bridging
    Hub-bridging ratio and DSpar separation metrics
network_properties
    Comprehensive network property measurements
distance_metrics
    MMD, Wasserstein, and other distance metrics for comparison
"""

from .hub_bridging import (
    compute_hub_bridging_ratio,
    compute_dspar_separation,
    compute_dspar_score,
    classify_edges_by_hub_bridging,
    compute_hub_bridging_profile,
    _parse_communities,
    _partition_edges,
)
from .network_properties import (
    comprehensive_network_properties,
    compute_degree_distribution_stats,
    compute_rich_club_coefficient,
    compute_participation_coefficient,
    compute_within_module_degree,
)
from .distance_metrics import (
    maximum_mean_discrepancy,
    wasserstein_distance_1d,
    ks_distance,
    property_distance_vector,
)

__all__ = [
    # Hub-bridging metrics
    "compute_hub_bridging_ratio",
    "compute_dspar_separation",
    "compute_dspar_score",
    "classify_edges_by_hub_bridging",
    # Network properties
    "comprehensive_network_properties",
    "compute_degree_distribution_stats",
    "compute_rich_club_coefficient",
    "compute_participation_coefficient",
    "compute_within_module_degree",
    # Distance metrics
    "maximum_mean_discrepancy",
    "wasserstein_distance_1d",
    "ks_distance",
    "property_distance_vector",
]
