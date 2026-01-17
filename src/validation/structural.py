"""
Structural Validation Module
============================

This module implements Experiments 1-4 for validating structural
properties of hub-bridging generators:

- Experiment 1: Parameter control (h -> rho_HB)
- Experiment 2: Degree distribution preservation
- Experiment 3: Modularity independence from h
- Experiment 4: Concentration/variance analysis

References
----------
.. [1] Your PhD thesis or publication
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats

logger = logging.getLogger(__name__)


def experiment_1_parameter_control(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    n_samples: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 1: Validate that parameter h controls rho_HB.

    This experiment tests that:
    1. rho_HB increases monotonically with h
    2. The relationship is statistically significant
    3. A polynomial model fits the relationship well

    Parameters
    ----------
    generator_func : Callable
        Generator function (e.g., hb_lfr, hb_sbm)
    generator_params : Dict[str, Any]
        Base generator parameters (excluding 'h' and 'seed')
    h_values : List[float], optional
        Values of h to test.
        Default: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_samples : int, optional
        Number of samples per h value (default: 30)
    seed : int, optional
        Base random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'h_values': tested h values
        - 'rho_samples': rho_HB samples (n_h x n_samples)
        - 'rho_mean': mean rho_HB for each h
        - 'rho_std': std dev for each h
        - 'rho_ci_lower': lower 95% CI
        - 'rho_ci_upper': upper 95% CI
        - 'monotonicity_test': results from test_monotonicity
        - 'fit_results': polynomial fit results
        - 'spearman_correlation': Spearman correlation (h, rho)

    Examples
    --------
    >>> from hub_bridging_validation.generators import hb_lfr
    >>> params = {'n': 250, 'mu': 0.3}
    >>> results = experiment_1_parameter_control(hb_lfr, params, n_samples=10, seed=42)
    >>> results['monotonicity_test']['is_monotonic']
    True
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio
    from .statistical_tests import test_monotonicity

    if h_values is None:
        h_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 1: Testing {len(h_values)} h values with {n_samples} samples each"
    )

    n_h = len(h_values)
    rho_samples = np.zeros((n_h, n_samples))

    for i, h in enumerate(h_values):
        logger.info(f"  h = {h}")

        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, communities = generator_func(**params)
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples[i, j] = rho

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")
                rho_samples[i, j] = np.nan

    # Compute statistics
    rho_mean = np.nanmean(rho_samples, axis=1)
    rho_std = np.nanstd(rho_samples, axis=1)

    # 95% confidence intervals
    n_valid = np.sum(~np.isnan(rho_samples), axis=1)
    ci_multiplier = stats.t.ppf(0.975, n_valid - 1)
    rho_ci_lower = rho_mean - ci_multiplier * rho_std / np.sqrt(n_valid)
    rho_ci_upper = rho_mean + ci_multiplier * rho_std / np.sqrt(n_valid)

    # Test monotonicity
    monotonicity_test = test_monotonicity(h_values, rho_samples)

    # Polynomial fit
    valid_mask = ~np.isnan(rho_mean)
    if valid_mask.sum() >= 3:
        coeffs = np.polyfit(
            np.array(h_values)[valid_mask],
            rho_mean[valid_mask],
            deg=2,
        )
        fit_results = {
            "coefficients": coeffs.tolist(),
            "r_squared": _compute_r_squared(
                np.array(h_values)[valid_mask],
                rho_mean[valid_mask],
                coeffs,
            ),
        }
    else:
        fit_results = {"coefficients": None, "r_squared": np.nan}

    # Spearman correlation
    all_h = np.repeat(h_values, n_samples)
    all_rho = rho_samples.flatten()
    valid = ~np.isnan(all_rho)
    if valid.sum() > 2:
        spearman_r, spearman_p = stats.spearmanr(all_h[valid], all_rho[valid])
    else:
        spearman_r, spearman_p = np.nan, np.nan

    results = {
        "h_values": np.array(h_values),
        "rho_samples": rho_samples,
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "rho_ci_lower": rho_ci_lower,
        "rho_ci_upper": rho_ci_upper,
        "monotonicity_test": monotonicity_test,
        "fit_results": fit_results,
        "spearman_correlation": {"r": spearman_r, "p_value": spearman_p},
    }

    logger.info(
        f"Experiment 1 complete: monotonic={monotonicity_test['is_monotonic']}, "
        f"Spearman r={spearman_r:.4f}"
    )

    return results


def _compute_r_squared(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    coeffs: NDArray[np.float64],
) -> float:
    """Compute R-squared for polynomial fit."""
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


def experiment_2_degree_preservation(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    n_samples: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 2: Validate degree distribution preservation.

    This experiment tests that the degree distribution is preserved
    across different h values (since rewiring should not change degrees).

    Parameters
    ----------
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Base generator parameters
    h_values : List[float], optional
        Values of h to test. Default: [0.0, 0.5, 1.0]
    n_samples : int, optional
        Number of samples per h value (default: 30)
    seed : int, optional
        Base random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'h_values': tested h values
        - 'degree_stats': degree statistics for each h
        - 'ks_tests': KS test results comparing h>0 to h=0
        - 'power_law_exponents': fitted power-law exponents
        - 'preservation_passed': whether preservation tests pass
    """
    from ..metrics.network_properties import compute_degree_distribution_stats
    from .statistical_tests import validate_degree_preservation

    if h_values is None:
        h_values = [0.0, 0.5, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 2: Testing degree preservation for {len(h_values)} h values"
    )

    degree_samples: Dict[float, List[NDArray]] = {h: [] for h in h_values}
    degree_stats: Dict[float, List[Dict]] = {h: [] for h in h_values}

    for i, h in enumerate(h_values):
        logger.info(f"  h = {h}")

        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, _ = generator_func(**params)
                degrees = np.array([d for _, d in G.degree()])
                degree_samples[h].append(degrees)

                stats_dict = compute_degree_distribution_stats(G, fit_powerlaw=True)
                degree_stats[h].append(stats_dict)

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")

    # Compare degree distributions using KS test
    ks_tests = {}
    baseline_degrees = np.concatenate(degree_samples[h_values[0]])

    for h in h_values[1:]:
        if degree_samples[h]:
            test_degrees = np.concatenate(degree_samples[h])
            ks_stat, ks_p = stats.ks_2samp(baseline_degrees, test_degrees)
            ks_tests[h] = {"statistic": ks_stat, "p_value": ks_p}

    # Extract power-law exponents
    power_law_exponents = {}
    for h in h_values:
        exponents = [
            s.get("powerlaw_alpha", np.nan)
            for s in degree_stats[h]
            if not np.isnan(s.get("powerlaw_alpha", np.nan))
        ]
        if exponents:
            power_law_exponents[h] = {
                "mean": np.mean(exponents),
                "std": np.std(exponents),
            }

    # Determine if preservation passed (all KS p-values > 0.01)
    preservation_passed = all(
        test["p_value"] > 0.01 for test in ks_tests.values()
    )

    results = {
        "h_values": h_values,
        "degree_stats": {
            h: {
                "mean_degree": np.mean([s["mean"] for s in stats_list]),
                "std_degree": np.mean([s["std"] for s in stats_list]),
                "max_degree": np.mean([s["max"] for s in stats_list]),
            }
            for h, stats_list in degree_stats.items()
            if stats_list
        },
        "ks_tests": ks_tests,
        "power_law_exponents": power_law_exponents,
        "preservation_passed": preservation_passed,
    }

    logger.info(f"Experiment 2 complete: preservation_passed={preservation_passed}")

    return results


def experiment_3_modularity_independence(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    mu_values: Optional[List[float]] = None,
    n_samples: int = 30,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 3: Validate modularity independence from h.

    This experiment tests that the modularity Q is controlled by mu,
    not by h. That is, for fixed mu, Q should be similar across h values.

    Parameters
    ----------
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Base generator parameters (mu will be varied)
    h_values : List[float], optional
        Values of h to test. Default: [0.0, 0.5, 1.0]
    mu_values : List[float], optional
        Values of mu to test. Default: [0.1, 0.3, 0.5]
    n_samples : int, optional
        Number of samples per (h, mu) combination
    seed : int, optional
        Base random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary with modularity analysis results
    """
    from ..metrics.network_properties import compute_modularity

    if h_values is None:
        h_values = [0.0, 0.5, 1.0]
    if mu_values is None:
        mu_values = [0.1, 0.3, 0.5]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 3: Testing {len(h_values)} h values x {len(mu_values)} mu values"
    )

    # Store results as (mu, h) -> list of modularity values
    modularity_samples: Dict[Tuple[float, float], List[float]] = {}

    for mu in mu_values:
        for h in h_values:
            key = (mu, h)
            modularity_samples[key] = []

            for j in range(n_samples):
                sample_seed = int(rng.integers(0, 2**31))

                try:
                    params = generator_params.copy()
                    params["h"] = h
                    params["mu"] = mu
                    params["seed"] = sample_seed

                    G, communities = generator_func(**params)
                    Q = compute_modularity(G, communities)
                    modularity_samples[key].append(Q)

                except Exception as e:
                    logger.warning(f"Sample mu={mu}, h={h}, j={j} failed: {e}")

    # For each mu, test if h affects modularity (should NOT)
    independence_tests = {}
    for mu in mu_values:
        groups = [modularity_samples[(mu, h)] for h in h_values]
        # Filter out empty groups
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            # Kruskal-Wallis test (non-parametric ANOVA)
            stat, p_value = stats.kruskal(*groups)
            independence_tests[mu] = {
                "statistic": stat,
                "p_value": p_value,
                "independent": p_value > 0.05,  # h does NOT affect Q
            }

    # Compute mean modularity for each (mu, h)
    modularity_means = {
        key: np.mean(values) if values else np.nan
        for key, values in modularity_samples.items()
    }

    results = {
        "h_values": h_values,
        "mu_values": mu_values,
        "modularity_samples": {str(k): v for k, v in modularity_samples.items()},
        "modularity_means": {str(k): v for k, v in modularity_means.items()},
        "independence_tests": independence_tests,
        "all_independent": all(
            test["independent"] for test in independence_tests.values()
        ),
    }

    logger.info(
        f"Experiment 3 complete: all_independent={results['all_independent']}"
    )

    return results


def experiment_4_concentration(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Experiment 4: Analyze concentration/variance of rho_HB.

    This experiment tests that rho_HB has low variance (concentrated
    around mean) for each h value, indicating reliable generation.

    Parameters
    ----------
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Base generator parameters
    h_values : List[float], optional
        Values of h to test. Default: [0.0, 0.5, 1.0]
    n_samples : int, optional
        Number of samples per h value (default: 100)
    seed : int, optional
        Base random seed

    Returns
    -------
    Dict[str, Any]
        Dictionary with concentration analysis results:
        - 'h_values': tested h values
        - 'rho_samples': all samples
        - 'cv': coefficient of variation for each h
        - 'normality_tests': Shapiro-Wilk test results
        - 'is_concentrated': whether CV < threshold (0.1)
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    if h_values is None:
        h_values = [0.0, 0.5, 1.0]

    rng = np.random.default_rng(seed)

    logger.info(
        f"Experiment 4: Testing concentration with {n_samples} samples per h"
    )

    rho_samples: Dict[float, List[float]] = {h: [] for h in h_values}

    for h in h_values:
        logger.info(f"  h = {h}")

        for j in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, communities = generator_func(**params)
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples[h].append(rho)

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")

    # Compute coefficient of variation
    cv_results = {}
    for h in h_values:
        samples = np.array(rho_samples[h])
        if len(samples) > 0:
            cv_results[h] = float(np.std(samples) / np.mean(samples))
        else:
            cv_results[h] = np.nan

    # Normality tests
    normality_tests = {}
    for h in h_values:
        samples = np.array(rho_samples[h])
        if len(samples) >= 20:  # Need sufficient samples for Shapiro-Wilk
            stat, p_value = stats.shapiro(samples)
            normality_tests[h] = {
                "statistic": stat,
                "p_value": p_value,
                "is_normal": p_value > 0.05,
            }

    # Check concentration (CV < 0.1 is considered concentrated)
    cv_threshold = 0.1
    is_concentrated = {
        h: cv < cv_threshold for h, cv in cv_results.items() if not np.isnan(cv)
    }

    results = {
        "h_values": h_values,
        "rho_samples": rho_samples,
        "cv": cv_results,
        "normality_tests": normality_tests,
        "is_concentrated": is_concentrated,
        "cv_threshold": cv_threshold,
        "all_concentrated": all(is_concentrated.values()),
    }

    logger.info(f"Experiment 4 complete: all_concentrated={results['all_concentrated']}")

    return results
