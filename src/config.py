"""
Configuration constants for Hub-Bridging Validation Framework.
===============================================================

This module contains all configuration constants used throughout
the validation framework. Centralizing these ensures consistency
and reproducibility.
"""

# Random seed for all stochastic operations
RANDOM_SEED = 42

# Community detection configuration
COMMUNITY_DETECTION_METHOD = 'leiden'  # Always use Leiden (state-of-the-art)
COMMUNITY_DETECTION_SEED = RANDOM_SEED

# Leiden algorithm citation
# Traag, V.A., Waltman, L. & van Eck, N.J.
# From Louvain to Leiden: guaranteeing well-connected communities.
# Sci Rep 9, 5233 (2019). https://doi.org/10.1038/s41598-019-41695-z

# HB-LFR generator defaults
DEFAULT_MAX_ITERS = 5000
DEFAULT_TAU1 = 2.5
DEFAULT_TAU2 = 1.5
DEFAULT_MU = 0.3

# Experiment 5 defaults
EXP5_N_SYNTHETIC_PER_REAL = 30
EXP5_N_CALIBRATION_SAMPLES = 10
EXP5_N_H_POINTS = 25
EXP5_EXTENDED_H_RANGE = (-0.5, 3.5)

# File paths
DEFAULT_DATA_DIR = "data/real_networks"
DEFAULT_RESULTS_DIR = "data/results"
