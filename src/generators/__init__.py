"""
Generators Module
=================

This module provides hub-bridging graph generators (HB-LFR, HB-SBM)
and wrappers for standard benchmark generators.

Submodules
----------
hb_lfr
    Hub-Bridging LFR generator (rewiring-based)
hb_lfr_direct
    Hub-Bridging LFR generator (direct generation)
hb_sbm
    Hub-Bridging Stochastic Block Model generator
base_generators
    Standard LFR and SBM wrappers
calibration
    h -> rho_HB calibration utilities
"""

from .hb_lfr import hb_lfr, hb_lfr_rewiring
from .hb_lfr_direct import hb_lfr_direct
from .hb_sbm import hb_sbm
from .base_generators import (
    generate_lfr,
    generate_sbm,
    generate_planted_partition,
)
from .calibration import (
    calibrate_h_to_rho,
    get_calibration_curve,
    h_for_target_rho,
)

__all__ = [
    # HB-LFR
    "hb_lfr",
    "hb_lfr_rewiring",
    "hb_lfr_direct",
    # HB-SBM
    "hb_sbm",
    # Base generators
    "generate_lfr",
    "generate_sbm",
    "generate_planted_partition",
    # Calibration
    "calibrate_h_to_rho",
    "get_calibration_curve",
    "h_for_target_rho",
]
