"""
Data Loading Module
===================

This module provides utilities for loading and processing real network
datasets from various sources (SNAP, KONECT, etc.).

Community Detection
-------------------
All experiments use the Leiden algorithm for community detection:
- Paper: Traag et al. (2019) "From Louvain to Leiden"
- Why: State-of-the-art, fixes Louvain's disconnected communities issue
- Reproducibility: Fixed random seed (42) ensures consistent results
"""

from .network_loader import (
    load_real_networks_from_snap,
    detect_communities_if_missing,
    detect_communities_robust,
    detect_communities_leiden,
    validate_communities,
    extract_network_metadata,
    load_network_file,
    create_sample_networks,
    get_expected_rho,
    load_networks_for_experiment_5,
    EXPECTED_RHO_HB,
)

__all__ = [
    "load_real_networks_from_snap",
    "detect_communities_if_missing",
    "detect_communities_robust",
    "detect_communities_leiden",
    "validate_communities",
    "extract_network_metadata",
    "load_network_file",
    "create_sample_networks",
    "get_expected_rho",
    "load_networks_for_experiment_5",
    "EXPECTED_RHO_HB",
]
