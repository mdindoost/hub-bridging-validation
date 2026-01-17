"""
Hub-Bridging Validation Framework
=================================

A comprehensive validation framework for hub-bridging graph generators
(HB-LFR and HB-SBM) developed for PhD research.

This framework validates that these generators:
1. Correctly control hub-bridging ratio (rho_HB) via parameter h
2. Match real network properties better than standard benchmarks
3. Replicate algorithmic behaviors observed on real networks

Modules
-------
metrics
    Hub-bridging metrics, network properties, and distance metrics
generators
    HB-LFR, HB-SBM, and standard benchmark generators
validation
    Structural, realism, and algorithmic validation experiments
algorithms
    Community detection and sparsification algorithms
visualization
    Plotting and table generation utilities
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@university.edu"

from . import metrics
from . import generators
from . import validation
from . import algorithms

# Optional visualization import (requires seaborn)
try:
    from . import visualization
    _has_visualization = True
except ImportError:
    visualization = None
    _has_visualization = False

__all__ = [
    "metrics",
    "generators",
    "validation",
    "algorithms",
    "visualization",
    "__version__",
]
