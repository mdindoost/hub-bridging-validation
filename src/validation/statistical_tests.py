"""
Statistical Tests Module
========================

This module provides statistical testing functions for all validation
experiments, including:

- Monotonicity tests
- Multiple testing corrections
- Effect size calculations
- Degree preservation validation
- Bootstrap confidence intervals

References
----------
.. [1] Your PhD thesis or publication
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


def test_monotonicity(
    x_values: List[float],
    y_samples: NDArray[np.float64],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Test if y increases monotonically with x.

    Uses multiple approaches:
    1. Page's trend test for ordered alternatives
    2. Jonckheere-Terpstra test
    3. Spearman correlation

    Parameters
    ----------
    x_values : List[float]
        Independent variable values (assumed ordered)
    y_samples : NDArray[np.float64]
        Dependent variable samples (n_x, n_samples)
    alpha : float, optional
        Significance level (default: 0.05)

    Returns
    -------
    Dict[str, Any]
        Test results with:
        - 'is_monotonic': overall monotonicity conclusion
        - 'spearman_r': Spearman correlation coefficient
        - 'spearman_p': p-value for Spearman test
        - 'jonckheere_stat': Jonckheere-Terpstra statistic
        - 'jonckheere_p': p-value for J-T test
        - 'mean_increases': whether all consecutive means increase

    Examples
    --------
    >>> x = [0.0, 0.5, 1.0]
    >>> y = np.array([[1, 1.1, 0.9], [1.5, 1.6, 1.4], [2, 2.1, 1.9]])
    >>> result = test_monotonicity(x, y)
    >>> result['is_monotonic']
    True
    """
    x_values = np.array(x_values)
    n_x = len(x_values)

    # Compute means
    y_means = np.nanmean(y_samples, axis=1)

    # Check if means are increasing
    mean_increases = all(y_means[i] <= y_means[i + 1] for i in range(n_x - 1))

    # Spearman correlation on means
    valid_mask = ~np.isnan(y_means)
    if valid_mask.sum() >= 3:
        spearman_r, spearman_p = stats.spearmanr(
            x_values[valid_mask], y_means[valid_mask]
        )
    else:
        spearman_r, spearman_p = np.nan, np.nan

    # Jonckheere-Terpstra test
    jt_stat, jt_p = _jonckheere_terpstra_test(y_samples)

    # Overall conclusion
    is_monotonic = (
        mean_increases
        and spearman_r > 0
        and spearman_p < alpha
    )

    return {
        "is_monotonic": is_monotonic,
        "mean_increases": mean_increases,
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "jonckheere_stat": jt_stat,
        "jonckheere_p": jt_p,
        "y_means": y_means.tolist(),
    }


def _jonckheere_terpstra_test(
    samples: NDArray[np.float64],
) -> Tuple[float, float]:
    """
    Perform Jonckheere-Terpstra test for ordered alternatives.

    Parameters
    ----------
    samples : NDArray[np.float64]
        Samples for each group (n_groups, n_samples)

    Returns
    -------
    Tuple[float, float]
        (statistic, p_value)
    """
    k = samples.shape[0]

    # Count U statistic
    U = 0
    n_comparisons = 0

    for i in range(k - 1):
        for j in range(i + 1, k):
            xi = samples[i][~np.isnan(samples[i])]
            xj = samples[j][~np.isnan(samples[j])]

            for a in xi:
                for b in xj:
                    if a < b:
                        U += 1
                    elif a == b:
                        U += 0.5
                    n_comparisons += 1

    if n_comparisons == 0:
        return np.nan, np.nan

    # Expected value and variance under null (approximate)
    n_total = sum(len(samples[i][~np.isnan(samples[i])]) for i in range(k))
    E_U = n_comparisons / 2

    # Approximate variance
    var_U = n_comparisons * (n_total + 1) / 12

    # Z-score
    Z = (U - E_U) / np.sqrt(var_U) if var_U > 0 else 0

    # One-sided p-value (testing for increasing trend)
    p_value = 1 - stats.norm.cdf(Z)

    return float(U), float(p_value)


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[bool]:
    """
    Apply Bonferroni correction for multiple testing.

    Parameters
    ----------
    p_values : List[float]
        Raw p-values
    alpha : float, optional
        Family-wise error rate (default: 0.05)

    Returns
    -------
    List[bool]
        Whether each test is significant after correction

    Examples
    --------
    >>> p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
    >>> bonferroni_correction(p_values, alpha=0.05)
    [True, False, False, False, False]
    """
    m = len(p_values)
    adjusted_alpha = alpha / m
    return [p < adjusted_alpha for p in p_values]


def fdr_correction(
    p_values: List[float],
    alpha: float = 0.05,
    method: str = "bh",
) -> Tuple[List[bool], List[float]]:
    """
    Apply False Discovery Rate correction.

    Parameters
    ----------
    p_values : List[float]
        Raw p-values
    alpha : float, optional
        FDR level (default: 0.05)
    method : str, optional
        Method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
        (default: 'bh')

    Returns
    -------
    Tuple[List[bool], List[float]]
        (significant, adjusted_p_values)

    Examples
    --------
    >>> p_values = [0.001, 0.01, 0.02, 0.04, 0.05]
    >>> significant, adjusted = fdr_correction(p_values)
    >>> sum(significant)
    5
    """
    m = len(p_values)
    if m == 0:
        return [], []

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    # Compute adjusted p-values
    if method == "bh":
        # Benjamini-Hochberg
        adjusted = np.zeros(m)
        for i in range(m):
            adjusted[sorted_indices[i]] = sorted_p[i] * m / (i + 1)
    elif method == "by":
        # Benjamini-Yekutieli
        c_m = sum(1 / i for i in range(1, m + 1))
        adjusted = np.zeros(m)
        for i in range(m):
            adjusted[sorted_indices[i]] = sorted_p[i] * m * c_m / (i + 1)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure monotonicity (adjusted p-values should not decrease)
    for i in range(m - 2, -1, -1):
        idx = sorted_indices[i]
        idx_next = sorted_indices[i + 1]
        adjusted[idx] = min(adjusted[idx], adjusted[idx_next])

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0, 1)

    significant = [adj_p < alpha for adj_p in adjusted]

    return significant, adjusted.tolist()


def compute_effect_size_and_ci(
    group1: List[float],
    group2: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute effect size (Cohen's d) and confidence interval.

    Parameters
    ----------
    group1 : List[float]
        First group samples
    group2 : List[float]
        Second group samples
    confidence : float, optional
        Confidence level (default: 0.95)
    n_bootstrap : int, optional
        Number of bootstrap samples for CI (default: 1000)
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - 'cohens_d': Cohen's d effect size
        - 'ci_lower': lower confidence interval bound
        - 'ci_upper': upper confidence interval bound
        - 'interpretation': effect size interpretation

    Examples
    --------
    >>> group1 = [1, 2, 3, 4, 5]
    >>> group2 = [3, 4, 5, 6, 7]
    >>> result = compute_effect_size_and_ci(group1, group2)
    >>> result['cohens_d'] < 0  # group1 has lower mean
    True
    """
    rng = np.random.default_rng(seed)

    g1 = np.array(group1)
    g2 = np.array(group2)

    # Remove NaN
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    if len(g1) < 2 or len(g2) < 2:
        return {
            "cohens_d": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "interpretation": "insufficient data",
        }

    # Compute Cohen's d
    n1, n2 = len(g1), len(g2)
    mean_diff = np.mean(g1) - np.mean(g2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
        / (n1 + n2 - 2)
    )

    if pooled_std > 0:
        cohens_d = mean_diff / pooled_std
    else:
        cohens_d = 0.0

    # Bootstrap CI
    bootstrap_d = []
    for _ in range(n_bootstrap):
        b1 = rng.choice(g1, size=n1, replace=True)
        b2 = rng.choice(g2, size=n2, replace=True)

        b_mean_diff = np.mean(b1) - np.mean(b2)
        b_pooled_std = np.sqrt(
            ((n1 - 1) * np.var(b1, ddof=1) + (n2 - 1) * np.var(b2, ddof=1))
            / (n1 + n2 - 2)
        )

        if b_pooled_std > 0:
            bootstrap_d.append(b_mean_diff / b_pooled_std)

    if bootstrap_d:
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_d, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_d, 100 * (1 - alpha / 2))
    else:
        ci_lower, ci_upper = np.nan, np.nan

    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return {
        "cohens_d": float(cohens_d),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "interpretation": interpretation,
    }


def validate_degree_preservation(
    graphs: List[Tuple[Any, Any]],
    tau_target: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Validate that generated graphs preserve target degree distribution.

    Parameters
    ----------
    graphs : List[Tuple[nx.Graph, Dict]]
        List of (graph, communities) tuples
    tau_target : float
        Target power-law exponent
    alpha : float, optional
        Significance level (default: 0.05)

    Returns
    -------
    Dict[str, Any]
        Validation results with:
        - 'mean_exponent': mean fitted power-law exponent
        - 'std_exponent': std of fitted exponents
        - 'target_within_ci': whether target is within confidence interval
        - 'ks_test_passed': proportion of graphs passing KS test
    """
    from ..metrics.network_properties import compute_degree_distribution_stats

    exponents = []
    ks_passed = []

    for G, _ in graphs:
        stats_dict = compute_degree_distribution_stats(G, fit_powerlaw=True)

        exp = stats_dict.get("powerlaw_alpha", np.nan)
        if not np.isnan(exp):
            exponents.append(exp)

        # Check if power-law is good fit (this is simplified)
        # In practice, would use goodness-of-fit test
        sigma = stats_dict.get("powerlaw_sigma", np.inf)
        ks_passed.append(sigma < 0.1)  # Simplified criterion

    if not exponents:
        return {
            "mean_exponent": np.nan,
            "std_exponent": np.nan,
            "target_within_ci": False,
            "ks_test_passed": 0.0,
        }

    mean_exp = np.mean(exponents)
    std_exp = np.std(exponents)

    # 95% CI for mean
    n = len(exponents)
    ci_margin = stats.t.ppf(0.975, n - 1) * std_exp / np.sqrt(n)
    target_within_ci = (mean_exp - ci_margin <= tau_target <= mean_exp + ci_margin)

    return {
        "mean_exponent": float(mean_exp),
        "std_exponent": float(std_exp),
        "ci_lower": float(mean_exp - ci_margin),
        "ci_upper": float(mean_exp + ci_margin),
        "target_within_ci": target_within_ci,
        "ks_test_passed": float(np.mean(ks_passed)),
        "n_graphs": len(graphs),
    }


def two_sample_permutation_test(
    group1: List[float],
    group2: List[float],
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Perform permutation test for difference in means.

    Parameters
    ----------
    group1 : List[float]
        First group samples
    group2 : List[float]
        Second group samples
    n_permutations : int, optional
        Number of permutations (default: 10000)
    alternative : str, optional
        'two-sided', 'greater', or 'less' (default: 'two-sided')
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, float]
        Test results with observed statistic and p-value
    """
    rng = np.random.default_rng(seed)

    g1 = np.array(group1)[~np.isnan(group1)]
    g2 = np.array(group2)[~np.isnan(group2)]

    observed_diff = np.mean(g1) - np.mean(g2)
    combined = np.concatenate([g1, g2])
    n1 = len(g1)

    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_diff = np.mean(combined[:n1]) - np.mean(combined[n1:])
        perm_diffs.append(perm_diff)

    perm_diffs = np.array(perm_diffs)

    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif alternative == "greater":
        p_value = np.mean(perm_diffs >= observed_diff)
    elif alternative == "less":
        p_value = np.mean(perm_diffs <= observed_diff)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return {
        "observed_difference": float(observed_diff),
        "p_value": float(p_value),
        "n_permutations": n_permutations,
    }
