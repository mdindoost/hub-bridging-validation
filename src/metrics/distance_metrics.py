"""
Distance Metrics Module
=======================

This module provides distance metrics for comparing distributions
and network properties between synthetic and real networks.

Includes Maximum Mean Discrepancy (MMD), Wasserstein distance,
Kolmogorov-Smirnov statistics, and composite property distance vectors.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel

logger = logging.getLogger(__name__)


def maximum_mean_discrepancy(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.

    MMD is a kernel-based distance measure between probability distributions.
    It measures the distance between mean embeddings in a reproducing kernel
    Hilbert space.

    Parameters
    ----------
    X : NDArray[np.float64]
        Samples from first distribution (n_samples_1, n_features) or (n_samples_1,)
    Y : NDArray[np.float64]
        Samples from second distribution (n_samples_2, n_features) or (n_samples_2,)
    kernel : str, optional
        Kernel type: 'rbf' (default) or 'linear'
    gamma : float, optional
        Kernel bandwidth for RBF kernel. If None, uses median heuristic.

    Returns
    -------
    float
        MMD distance (always non-negative)

    Examples
    --------
    >>> X = np.random.normal(0, 1, 100)
    >>> Y = np.random.normal(0, 1, 100)
    >>> mmd = maximum_mean_discrepancy(X, Y)
    >>> mmd >= 0
    True
    >>> Y_shifted = np.random.normal(5, 1, 100)
    >>> mmd_shifted = maximum_mean_discrepancy(X, Y_shifted)
    >>> mmd_shifted > mmd
    True

    Notes
    -----
    The MMD is computed as:
        MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]

    where k is the kernel function.
    """
    # Ensure 2D arrays
    X = np.atleast_2d(X.reshape(-1, 1) if X.ndim == 1 else X)
    Y = np.atleast_2d(Y.reshape(-1, 1) if Y.ndim == 1 else Y)

    if gamma is None:
        # Median heuristic for bandwidth
        combined = np.vstack([X, Y])
        pairwise_dists = cdist(combined, combined, metric="euclidean")
        gamma = 1.0 / (2 * np.median(pairwise_dists[pairwise_dists > 0]) ** 2)

    if kernel == "rbf":
        K_XX = rbf_kernel(X, X, gamma=gamma)
        K_YY = rbf_kernel(Y, Y, gamma=gamma)
        K_XY = rbf_kernel(X, Y, gamma=gamma)
    elif kernel == "linear":
        K_XX = X @ X.T
        K_YY = Y @ Y.T
        K_XY = X @ Y.T
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    n_x = len(X)
    n_y = len(Y)

    # Unbiased estimator
    # Remove diagonal for XX and YY terms
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    mmd_squared = (
        K_XX.sum() / (n_x * (n_x - 1))
        + K_YY.sum() / (n_y * (n_y - 1))
        - 2 * K_XY.sum() / (n_x * n_y)
    )

    # Return MMD (take sqrt, handle numerical issues)
    return float(np.sqrt(max(0, mmd_squared)))


def wasserstein_distance_1d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> float:
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance).

    Parameters
    ----------
    X : NDArray[np.float64]
        Samples from first distribution
    Y : NDArray[np.float64]
        Samples from second distribution

    Returns
    -------
    float
        Wasserstein-1 distance

    Examples
    --------
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> Y = np.array([2, 3, 4, 5, 6])
    >>> w = wasserstein_distance_1d(X, Y)
    >>> abs(w - 1.0) < 0.01
    True
    """
    return float(stats.wasserstein_distance(X, Y))


def ks_distance(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Compute Kolmogorov-Smirnov statistic and p-value.

    The KS statistic measures the maximum distance between the
    empirical CDFs of two samples.

    Parameters
    ----------
    X : NDArray[np.float64]
        Samples from first distribution
    Y : NDArray[np.float64]
        Samples from second distribution

    Returns
    -------
    Tuple[float, float]
        (ks_statistic, p_value)

    Examples
    --------
    >>> X = np.random.normal(0, 1, 100)
    >>> Y = np.random.normal(0, 1, 100)
    >>> ks, p = ks_distance(X, Y)
    >>> 0 <= ks <= 1
    True
    >>> p > 0.01  # Same distribution, expect high p-value
    True
    """
    statistic, pvalue = stats.ks_2samp(X, Y)
    return float(statistic), float(pvalue)


def jensen_shannon_divergence(
    P: NDArray[np.float64],
    Q: NDArray[np.float64],
    base: float = 2.0,
) -> float:
    """
    Compute Jensen-Shannon divergence between two probability distributions.

    JSD is a symmetric, bounded measure of similarity between distributions.

    Parameters
    ----------
    P : NDArray[np.float64]
        First probability distribution (must sum to 1)
    Q : NDArray[np.float64]
        Second probability distribution (must sum to 1)
    base : float, optional
        Logarithm base (default: 2, giving bits)

    Returns
    -------
    float
        Jensen-Shannon divergence (0 = identical, 1 = maximally different for base=2)

    Examples
    --------
    >>> P = np.array([0.5, 0.5])
    >>> Q = np.array([0.5, 0.5])
    >>> jensen_shannon_divergence(P, Q)
    0.0
    """
    # Normalize to ensure valid probability distributions
    P = P / P.sum()
    Q = Q / Q.sum()

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    P = P + eps
    Q = Q + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    M = 0.5 * (P + Q)

    return float(0.5 * (stats.entropy(P, M, base=base) + stats.entropy(Q, M, base=base)))


def histogram_distance(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    bins: Union[int, str] = "auto",
    method: str = "wasserstein",
) -> float:
    """
    Compute distance between histograms of two samples.

    Parameters
    ----------
    X : NDArray[np.float64]
        Samples from first distribution
    Y : NDArray[np.float64]
        Samples from second distribution
    bins : int or str, optional
        Number of bins or binning strategy (default: 'auto')
    method : str, optional
        Distance method: 'wasserstein', 'jsd', or 'intersection'
        (default: 'wasserstein')

    Returns
    -------
    float
        Histogram distance
    """
    # Determine common bin edges
    combined = np.concatenate([X, Y])
    _, bin_edges = np.histogram(combined, bins=bins)

    # Compute histograms with same bins
    hist_X, _ = np.histogram(X, bins=bin_edges, density=True)
    hist_Y, _ = np.histogram(Y, bins=bin_edges, density=True)

    # Normalize to probability distributions
    hist_X = hist_X / (hist_X.sum() + 1e-10)
    hist_Y = hist_Y / (hist_Y.sum() + 1e-10)

    if method == "wasserstein":
        # Approximate Wasserstein using histogram
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return float(stats.wasserstein_distance(bin_centers, bin_centers, hist_X, hist_Y))
    elif method == "jsd":
        return jensen_shannon_divergence(hist_X, hist_Y)
    elif method == "intersection":
        return 1.0 - float(np.minimum(hist_X, hist_Y).sum())
    else:
        raise ValueError(f"Unknown method: {method}")


def property_distance_vector(
    props1: Dict[str, Any],
    props2: Dict[str, Any],
    property_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute distance vector between two sets of network properties.

    This function compares comprehensive network properties and returns
    distances for each property type.

    Parameters
    ----------
    props1 : Dict[str, Any]
        Properties of first network (from comprehensive_network_properties)
    props2 : Dict[str, Any]
        Properties of second network
    property_weights : Dict[str, float], optional
        Weights for combining distances (default: uniform)

    Returns
    -------
    Dict[str, float]
        Dictionary with distance for each property type and weighted total

    Examples
    --------
    >>> props1 = {'basic': {'n_nodes': 100}, 'degree': {'mean': 10}}
    >>> props2 = {'basic': {'n_nodes': 100}, 'degree': {'mean': 15}}
    >>> dists = property_distance_vector(props1, props2)
    >>> 'degree_mean' in dists
    True
    """
    distances: Dict[str, float] = {}

    # Define default weights
    if property_weights is None:
        property_weights = {
            "degree_distribution": 1.0,
            "clustering": 1.0,
            "path_length": 1.0,
            "modularity": 1.0,
            "hub_bridging": 1.0,
        }

    # Compare degree distributions
    if "degree" in props1 and "degree" in props2:
        deg1 = props1["degree"].get("degrees", np.array([]))
        deg2 = props2["degree"].get("degrees", np.array([]))

        if len(deg1) > 0 and len(deg2) > 0:
            distances["degree_mmd"] = maximum_mean_discrepancy(deg1, deg2)
            distances["degree_wasserstein"] = wasserstein_distance_1d(deg1, deg2)
            ks, _ = ks_distance(deg1, deg2)
            distances["degree_ks"] = ks

        # Compare summary statistics
        for stat in ["mean", "std", "skewness", "gini"]:
            if stat in props1["degree"] and stat in props2["degree"]:
                v1 = props1["degree"][stat]
                v2 = props2["degree"][stat]
                if not (np.isnan(v1) or np.isnan(v2)):
                    # Relative difference
                    denom = max(abs(v1), abs(v2), 1e-10)
                    distances[f"degree_{stat}"] = abs(v1 - v2) / denom

    # Compare clustering
    if "clustering" in props1 and "clustering" in props2:
        for stat in ["global", "average"]:
            if stat in props1["clustering"] and stat in props2["clustering"]:
                v1 = props1["clustering"][stat]
                v2 = props2["clustering"][stat]
                distances[f"clustering_{stat}"] = abs(v1 - v2)

    # Compare path lengths
    if "path_length" in props1 and "path_length" in props2:
        v1 = props1["path_length"].get("average_path_length", np.nan)
        v2 = props2["path_length"].get("average_path_length", np.nan)
        if not (np.isnan(v1) or np.isnan(v2)):
            denom = max(v1, v2, 1e-10)
            distances["path_length"] = abs(v1 - v2) / denom

    # Compare modularity
    if "community" in props1 and "community" in props2:
        v1 = props1["community"].get("modularity", np.nan)
        v2 = props2["community"].get("modularity", np.nan)
        if not (np.isnan(v1) or np.isnan(v2)):
            distances["modularity"] = abs(v1 - v2)

    # Compare hub-bridging
    if "hub_bridging" in props1 and "hub_bridging" in props2:
        v1 = props1["hub_bridging"].get("rho_hb", np.nan)
        v2 = props2["hub_bridging"].get("rho_hb", np.nan)
        if not (np.isnan(v1) or np.isnan(v2)):
            # Log-scale comparison for ratio
            distances["rho_hb"] = abs(np.log(v1 + 1e-10) - np.log(v2 + 1e-10))

    # Compute weighted total
    total = 0.0
    weight_sum = 0.0
    for key, weight in property_weights.items():
        matching_keys = [k for k in distances if key in k]
        if matching_keys:
            avg_dist = np.mean([distances[k] for k in matching_keys])
            total += weight * avg_dist
            weight_sum += weight

    if weight_sum > 0:
        distances["weighted_total"] = total / weight_sum
    else:
        distances["weighted_total"] = 0.0

    return distances


def compare_degree_distributions(
    degrees1: NDArray[np.int64],
    degrees2: NDArray[np.int64],
) -> Dict[str, float]:
    """
    Comprehensive comparison of two degree distributions.

    Parameters
    ----------
    degrees1 : NDArray[np.int64]
        Degree sequence of first graph
    degrees2 : NDArray[np.int64]
        Degree sequence of second graph

    Returns
    -------
    Dict[str, float]
        Dictionary with multiple distance metrics
    """
    d1 = degrees1.astype(float)
    d2 = degrees2.astype(float)

    ks_stat, ks_p = ks_distance(d1, d2)

    return {
        "mmd": maximum_mean_discrepancy(d1, d2),
        "wasserstein": wasserstein_distance_1d(d1, d2),
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_p,
        "mean_diff": abs(np.mean(d1) - np.mean(d2)),
        "std_diff": abs(np.std(d1) - np.std(d2)),
        "max_diff": abs(np.max(d1) - np.max(d2)),
    }
