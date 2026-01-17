"""
Validation Module
=================

This module provides validation experiments for hub-bridging generators.

Submodules
----------
structural
    Experiments 1-4: Parameter control, degree preservation, etc.
realism
    Experiments 5-6: Property matching, network fitting
algorithmic
    Experiments 7-8: Community detection, sparsification
statistical_tests
    Statistical testing functions for all experiments
"""

from .structural import (
    experiment_1_parameter_control,
    experiment_2_degree_preservation,
    experiment_3_modularity_independence,
    experiment_4_concentration,
)
from .realism import (
    experiment_5_property_matching,
    experiment_6_fitting,
)
from .algorithmic import (
    experiment_7_community_detection,
    experiment_8_sparsification,
)
from .statistical_tests import (
    test_monotonicity,
    bonferroni_correction,
    fdr_correction,
    compute_effect_size_and_ci,
    validate_degree_preservation,
)

__all__ = [
    # Structural validation
    "experiment_1_parameter_control",
    "experiment_2_degree_preservation",
    "experiment_3_modularity_independence",
    "experiment_4_concentration",
    # Realism validation
    "experiment_5_property_matching",
    "experiment_6_fitting",
    # Algorithmic validation
    "experiment_7_community_detection",
    "experiment_8_sparsification",
    # Statistical tests
    "test_monotonicity",
    "bonferroni_correction",
    "fdr_correction",
    "compute_effect_size_and_ci",
    "validate_degree_preservation",
]
