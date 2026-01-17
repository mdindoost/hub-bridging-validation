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


def experiment_2_degree_preservation_full(
    generators: List[str] = None,
    h_values: List[float] = None,
    n_samples: int = 20,
    n: int = 1000,
    target_tau1: float = 2.5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Experiment 2: Comprehensive degree distribution preservation validation.

    Tests that both HB-SBM and HB-LFR preserve power-law degree distributions
    across different h values, proving hub-bridging control is independent
    of degree structure.

    Parameters
    ----------
    generators : list
        Which generators to test ['hb_sbm', 'hb_lfr']
    h_values : list
        Hub-bridging parameters to test
    n_samples : int
        Samples per (generator, h) combination
    n : int
        Network size
    target_tau1 : float
        Target power-law exponent (typically 2.5)
    seed : int
        Random seed

    Returns
    -------
    dict
        Results for each generator with statistical tests:
        - tau_estimates: power-law exponent estimates
        - ks_statistics: goodness-of-fit statistics
        - anova_p: p-value for τ independence of h
        - passes: overall validation status
    """
    try:
        from powerlaw import Fit
        HAS_POWERLAW = True
    except ImportError:
        logger.warning("powerlaw package not installed, using fallback method")
        HAS_POWERLAW = False

    from ..generators.hb_sbm import hb_sbm
    from ..generators.hb_lfr import hb_lfr

    if generators is None:
        generators = ['hb_sbm', 'hb_lfr']
    if h_values is None:
        h_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: Degree Distribution Preservation")
    logger.info("=" * 70)
    logger.info(f"Generators: {generators}")
    logger.info(f"h values: {h_values}")
    logger.info(f"Samples per h: {n_samples}, n={n}")

    results = {}

    for gen_name in generators:
        logger.info(f"\n--- Testing {gen_name.upper()} ---")

        gen_results = {
            'h_values': h_values,
            'tau_estimates': {h: [] for h in h_values},
            'ks_statistics': {h: [] for h in h_values},
            'degree_sequences': {h: [] for h in h_values},
            'mean_degrees': {h: [] for h in h_values},
            'max_degrees': {h: [] for h in h_values},
        }

        for h in h_values:
            logger.info(f"\nGenerating graphs with h={h:.2f}...")

            for sample in range(n_samples):
                sample_seed = int(rng.integers(0, 2**31))

                try:
                    # Generate graph
                    if gen_name == 'hb_sbm':
                        G, communities = hb_sbm(
                            n=n, k=5, h=h, seed=sample_seed,
                            p_in=0.3, p_out=0.05,
                            theta_distribution='power_law',
                            degree_correction_scale=1.5
                        )
                    elif gen_name == 'hb_lfr':
                        G, communities = hb_lfr(
                            n=n, tau1=target_tau1, h=h, seed=sample_seed,
                            max_iters=3000
                        )
                    else:
                        raise ValueError(f"Unknown generator: {gen_name}")

                    # Extract degrees
                    degrees = [d for n_id, d in G.degree() if d > 0]
                    gen_results['degree_sequences'][h].append(degrees)
                    gen_results['mean_degrees'][h].append(np.mean(degrees))
                    gen_results['max_degrees'][h].append(np.max(degrees))

                    # Fit power-law
                    if HAS_POWERLAW and len(degrees) > 10:
                        try:
                            fit = Fit(degrees, discrete=True, verbose=False)
                            tau_est = fit.alpha
                            gen_results['tau_estimates'][h].append(tau_est)

                            # KS test: power-law vs exponential
                            R, p_val = fit.distribution_compare(
                                'power_law', 'exponential'
                            )
                            gen_results['ks_statistics'][h].append(R)

                        except Exception as e:
                            logger.debug(f"Power-law fit failed: {e}")
                            gen_results['tau_estimates'][h].append(np.nan)
                            gen_results['ks_statistics'][h].append(np.nan)
                    else:
                        # Fallback: estimate tau from log-log regression
                        try:
                            unique, counts = np.unique(degrees, return_counts=True)
                            valid = unique > 0
                            if valid.sum() > 3:
                                log_k = np.log(unique[valid])
                                log_p = np.log(counts[valid] / counts[valid].sum())
                                slope, _ = np.polyfit(log_k, log_p, 1)
                                tau_est = -slope
                                gen_results['tau_estimates'][h].append(tau_est)
                            else:
                                gen_results['tau_estimates'][h].append(np.nan)
                        except Exception:
                            gen_results['tau_estimates'][h].append(np.nan)
                        gen_results['ks_statistics'][h].append(np.nan)

                except Exception as e:
                    logger.warning(f"Sample h={h}, {sample} failed: {e}")
                    gen_results['tau_estimates'][h].append(np.nan)
                    gen_results['ks_statistics'][h].append(np.nan)

                if (sample + 1) % 5 == 0:
                    valid_tau = [t for t in gen_results['tau_estimates'][h] if not np.isnan(t)]
                    if valid_tau:
                        current_tau = np.mean(valid_tau)
                        logger.info(f"  Sample {sample+1}/{n_samples}: τ={current_tau:.3f}")

        # Statistical analysis
        logger.info(f"\n--- Statistical Analysis for {gen_name.upper()} ---")

        # 1. Mean τ by h value
        tau_means = {}
        tau_stds = {}
        for h in h_values:
            valid = [t for t in gen_results['tau_estimates'][h] if not np.isnan(t)]
            if valid:
                tau_means[h] = np.mean(valid)
                tau_stds[h] = np.std(valid)
            else:
                tau_means[h] = np.nan
                tau_stds[h] = np.nan

        logger.info("\nPower-law exponent τ by h:")
        for h in h_values:
            logger.info(f"  h={h:.2f}: τ={tau_means[h]:.3f} ± {tau_stds[h]:.3f}")

        # 2. Test H₀: τ independent of h (ANOVA)
        tau_samples_clean = []
        for h in h_values:
            valid = [t for t in gen_results['tau_estimates'][h] if not np.isnan(t)]
            if valid:
                tau_samples_clean.append(valid)

        if len(tau_samples_clean) >= 2 and all(len(s) > 0 for s in tau_samples_clean):
            f_stat, p_anova = stats.f_oneway(*tau_samples_clean)
            logger.info(f"\nANOVA test (H₀: τ independent of h):")
            logger.info(f"  F={f_stat:.3f}, p={p_anova:.4f}")

            independent = p_anova > 0.05
            logger.info(f"  Result: {'✓ Independent' if independent else '✗ Dependent'}")
        else:
            p_anova = np.nan
            independent = False
            logger.warning("Insufficient data for ANOVA")

        # 3. Test H₀: τ ≈ target_tau1 (t-test on all samples)
        all_tau = [t for h in h_values
                   for t in gen_results['tau_estimates'][h]
                   if not np.isnan(t)]

        if len(all_tau) > 2:
            t_stat, p_ttest = stats.ttest_1samp(all_tau, target_tau1)
            tau_overall_mean = np.mean(all_tau)
            tau_overall_std = np.std(all_tau)

            logger.info(f"\nt-test (H₀: τ={target_tau1}):")
            logger.info(f"  Overall τ: {tau_overall_mean:.3f} ± {tau_overall_std:.3f}")
            logger.info(f"  t={t_stat:.3f}, p={p_ttest:.4f}")

            close_to_target = abs(tau_overall_mean - target_tau1) < 0.3
            logger.info(f"  Result: {'✓ Close to target' if close_to_target else '✗ Deviates'}")
        else:
            p_ttest = np.nan
            tau_overall_mean = np.nan
            tau_overall_std = np.nan
            close_to_target = False

        # 4. KS test statistics (power-law goodness of fit)
        ks_means = {}
        for h in h_values:
            valid = [k for k in gen_results['ks_statistics'][h] if not np.isnan(k)]
            ks_means[h] = np.mean(valid) if valid else np.nan

        if HAS_POWERLAW:
            logger.info(f"\nKS statistics (power-law vs exponential):")
            for h in h_values:
                if not np.isnan(ks_means[h]):
                    logger.info(f"  h={h:.2f}: R={ks_means[h]:.3f} " +
                               "(positive = better power-law fit)")

        # Assessment
        passes = independent and close_to_target

        logger.info(f"\n{'='*50}")
        logger.info(f"VALIDATION ({gen_name.upper()}): {'PASS ✓' if passes else 'FAIL ✗'}")
        logger.info(f"{'='*50}")

        # Store results
        gen_results['statistics'] = {
            'tau_means': tau_means,
            'tau_stds': tau_stds,
            'tau_overall_mean': tau_overall_mean,
            'tau_overall_std': tau_overall_std,
            'anova_f': f_stat if 'f_stat' in dir() else np.nan,
            'anova_p': p_anova,
            'ttest_t': t_stat if 'p_ttest' in dir() and not np.isnan(p_ttest) else np.nan,
            'ttest_p': p_ttest,
            'independent': independent,
            'close_to_target': close_to_target,
            'passes': passes,
            'ks_means': ks_means,
            'target_tau1': target_tau1,
        }

        results[gen_name] = gen_results

    return results


def experiment_3_concentration(
    generators: List[str] = None,
    h_test: float = 1.0,
    n_samples: int = 100,
    n: int = 500,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Experiment 3: Concentration and Reproducibility.

    Tests that ρ_HB concentrates around expected value with low variance,
    proving generators are reliable for controlled experiments.

    Parameters
    ----------
    generators : list
        Which generators to test (default: ['hb_sbm', 'hb_lfr'])
    h_test : float
        Hub-bridging parameter to test (mid-range value)
    n_samples : int
        Number of independent generations (100 for robust statistics)
    n : int
        Network size
    k : int
        Number of communities
    seed : int
        Base random seed

    Returns
    -------
    dict
        Concentration statistics for each generator
    """
    import time

    from ..generators.hb_sbm import hb_sbm
    from ..generators.hb_lfr import hb_lfr
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    if generators is None:
        generators = ['hb_sbm', 'hb_lfr']

    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 3: Concentration and Reproducibility")
    logger.info("=" * 70)
    logger.info(f"Testing h={h_test:.2f} with {n_samples} samples per generator")
    logger.info(f"Network size: n={n}, communities: k={k}")

    results = {}

    for gen_name in generators:
        logger.info(f"\n--- Testing {gen_name.upper()} ---")

        rho_samples = []
        generation_times = []

        for sample in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))
            start_time = time.time()

            try:
                # Generate graph
                if gen_name == 'hb_sbm':
                    G, communities = hb_sbm(
                        n=n, k=k, h=h_test, seed=sample_seed,
                        p_in=0.3, p_out=0.05,
                        theta_distribution='power_law',
                        degree_correction_scale=1.5
                    )
                elif gen_name == 'hb_lfr':
                    G, communities = hb_lfr(
                        n=n, tau1=2.5, mu=0.3, h=h_test, seed=sample_seed,
                        max_iters=5000
                    )
                else:
                    raise ValueError(f"Unknown generator: {gen_name}")

                generation_time = time.time() - start_time
                generation_times.append(generation_time)

                # Measure ρ_HB
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples.append(rho)

                # Progress reporting
                if (sample + 1) % 20 == 0:
                    current_mean = np.mean(rho_samples)
                    current_std = np.std(rho_samples)
                    current_cv = (current_std / current_mean) * 100
                    logger.info(f"  Sample {sample+1}/{n_samples}: "
                               f"ρ={rho:.3f} (μ={current_mean:.3f}, σ={current_std:.3f}, CV={current_cv:.1f}%)")

            except Exception as e:
                logger.warning(f"  Sample {sample+1} failed: {e}")

        # Compute concentration statistics
        rho_array = np.array(rho_samples)
        rho_mean = np.mean(rho_array)
        rho_std = np.std(rho_array)
        rho_cv = rho_std / rho_mean  # Coefficient of variation

        # Confidence intervals
        ci_95 = 1.96 * rho_std / np.sqrt(len(rho_array))
        ci_99 = 2.576 * rho_std / np.sqrt(len(rho_array))

        # Percentiles
        rho_median = np.median(rho_array)
        rho_q25 = np.percentile(rho_array, 25)
        rho_q75 = np.percentile(rho_array, 75)
        rho_iqr = rho_q75 - rho_q25

        # Min/Max
        rho_min = np.min(rho_array)
        rho_max = np.max(rho_array)
        rho_range = rho_max - rho_min

        # Normality test (Shapiro-Wilk)
        if len(rho_array) >= 20:
            # Shapiro-Wilk limited to 5000 samples, use first 50 for speed
            shapiro_stat, shapiro_p = stats.shapiro(rho_array[:min(50, len(rho_array))])
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Test for outliers (using IQR method)
        lower_fence = rho_q25 - 1.5 * rho_iqr
        upper_fence = rho_q75 + 1.5 * rho_iqr
        outliers = [r for r in rho_array if r < lower_fence or r > upper_fence]
        n_outliers = len(outliers)

        # Generation time statistics
        mean_gen_time = np.mean(generation_times) if generation_times else 0
        std_gen_time = np.std(generation_times) if generation_times else 0
        total_gen_time = sum(generation_times)

        # Print detailed statistics
        logger.info(f"\n--- Concentration Statistics for {gen_name.upper()} ---")
        logger.info(f"Mean ρ_HB:        {rho_mean:.4f}")
        logger.info(f"Std dev:          {rho_std:.4f}")
        logger.info(f"Coefficient of variation (CV): {rho_cv:.2%}")
        logger.info(f"95% CI:           ±{ci_95:.4f}")
        logger.info(f"99% CI:           ±{ci_99:.4f}")
        logger.info(f"\nDistribution:")
        logger.info(f"  Median:         {rho_median:.4f}")
        logger.info(f"  25th percentile: {rho_q25:.4f}")
        logger.info(f"  75th percentile: {rho_q75:.4f}")
        logger.info(f"  IQR:            {rho_iqr:.4f}")
        logger.info(f"  Min:            {rho_min:.4f}")
        logger.info(f"  Max:            {rho_max:.4f}")
        logger.info(f"  Range:          {rho_range:.4f}")
        logger.info(f"\nDiagnostics:")
        logger.info(f"  Outliers:       {n_outliers}/{len(rho_array)} ({100*n_outliers/len(rho_array):.1f}%)")
        logger.info(f"  Shapiro-Wilk:   W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
        logger.info(f"  Distribution:   {'Normal' if shapiro_p > 0.05 else 'Non-normal'}")
        logger.info(f"\nPerformance:")
        logger.info(f"  Mean gen time:  {mean_gen_time:.3f}s ± {std_gen_time:.3f}s")
        logger.info(f"  Total time:     {total_gen_time:.1f}s")

        # Assessment
        passes_cv = rho_cv < 0.15  # CV < 15%
        passes_outliers = n_outliers / len(rho_array) < 0.05  # < 5% outliers
        passes_overall = passes_cv and passes_outliers

        logger.info(f"\n{'='*50}")
        logger.info(f"Assessment for {gen_name.upper()}:")
        logger.info(f"  CV < 15%:       {'✓ PASS' if passes_cv else '✗ FAIL'} (CV={rho_cv:.2%})")
        logger.info(f"  Outliers < 5%:  {'✓ PASS' if passes_outliers else '✗ FAIL'} ({100*n_outliers/len(rho_array):.1f}%)")
        logger.info(f"  Overall:        {'✓ PASS' if passes_overall else '✗ FAIL'}")
        logger.info(f"{'='*50}")

        # Store results
        results[gen_name] = {
            'h_test': h_test,
            'n_samples': len(rho_array),
            'rho_samples': rho_samples,
            'statistics': {
                'mean': rho_mean,
                'std': rho_std,
                'cv': rho_cv,
                'median': rho_median,
                'q25': rho_q25,
                'q75': rho_q75,
                'iqr': rho_iqr,
                'min': rho_min,
                'max': rho_max,
                'range': rho_range,
                'ci_95': ci_95,
                'ci_99': ci_99,
                'n_outliers': n_outliers,
                'outlier_rate': n_outliers / len(rho_array),
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'is_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else None
            },
            'performance': {
                'mean_time': mean_gen_time,
                'std_time': std_gen_time,
                'total_time': total_gen_time
            },
            'assessment': {
                'passes_cv': passes_cv,
                'passes_outliers': passes_outliers,
                'passes_overall': passes_overall
            }
        }

    # Comparative analysis
    logger.info("\n" + "=" * 70)
    logger.info("COMPARATIVE ANALYSIS")
    logger.info("=" * 70)

    logger.info(f"\n{'Generator':<12} | {'Mean ρ':>10} | {'Std':>8} | {'CV':>8} | {'95% CI':>10} | {'Outliers':>10} | {'Status':>10}")
    logger.info("-" * 80)

    for gen_name in generators:
        if gen_name in results:
            res = results[gen_name]
            s = res['statistics']
            a = res['assessment']
            status = '✓ PASS' if a['passes_overall'] else '✗ FAIL'
            logger.info(f"{gen_name.upper():<12} | {s['mean']:>10.4f} | {s['std']:>8.4f} | {s['cv']:>7.2%} | ±{s['ci_95']:>8.4f} | {s['n_outliers']:>4}/{res['n_samples']:<5} | {status:>10}")

    # Determine which is more precise
    if 'hb_sbm' in results and 'hb_lfr' in results:
        cv_sbm = results['hb_sbm']['statistics']['cv']
        cv_lfr = results['hb_lfr']['statistics']['cv']
        if cv_lfr > 0:
            precision_ratio = cv_sbm / cv_lfr
            logger.info(f"\nPrecision comparison:")
            if precision_ratio > 1:
                logger.info(f"  HB-LFR is {precision_ratio:.1f}x more precise than HB-SBM")
            else:
                logger.info(f"  HB-SBM is {1/precision_ratio:.1f}x more precise than HB-LFR")
            logger.info(f"  (lower CV = tighter concentration)")

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


def experiment_4_modularity_independence(
    generators: List[str] = None,
    h_values: List[float] = None,
    n_samples: int = 30,
    n: int = 500,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Experiment 4: Modularity Independence Test.

    Validates Theorem 4(a): For fixed constraints (degree sequence,
    community structure, |E_inter|), modularity Q is independent of
    hub-bridging ratio ρ_HB.

    This proves that hub-bridging control affects EDGE PLACEMENT,
    not the fundamental community structure quality.

    Parameters
    ----------
    generators : list
        Which generators to test. HB-LFR preferred for exact degree preservation.
    h_values : list
        Hub-bridging parameters to test
    n_samples : int
        Samples per h value
    n : int
        Network size
    k : int
        Number of communities
    seed : int
        Base random seed

    Returns
    -------
    dict
        Modularity statistics and independence tests
    """
    import networkx as nx
    from scipy.stats import pearsonr, spearmanr, f_oneway

    from ..generators.hb_sbm import hb_sbm
    from ..generators.hb_lfr import hb_lfr
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    if generators is None:
        generators = ['hb_lfr']  # HB-LFR preferred for exact degree preservation
    if h_values is None:
        h_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    rng = np.random.default_rng(seed)

    logger.info("=" * 70)
    logger.info("EXPERIMENT 4: Modularity Independence Test")
    logger.info("=" * 70)
    logger.info("Testing Theorem 4(a): Q independent of ρ_HB for fixed constraints")
    logger.info(f"Generators: {generators}")
    logger.info(f"h values: {h_values}")
    logger.info(f"Samples per h: {n_samples}")
    logger.info(f"Network size: n={n}, communities: k={k}")
    logger.info("")

    results = {}

    for gen_name in generators:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Testing {gen_name.upper()}")
        logger.info(f"{'=' * 70}")

        gen_results = {
            'h_values': h_values,
            'rho_samples': {h: [] for h in h_values},
            'Q_samples': {h: [] for h in h_values},
            'degree_sequences': {h: [] for h in h_values}
        }

        for h in h_values:
            logger.info(f"\n--- h = {h:.2f} ---")

            for sample in range(n_samples):
                sample_seed = int(rng.integers(0, 2**31))

                try:
                    # Generate graph
                    if gen_name == 'hb_sbm':
                        G, communities = hb_sbm(
                            n=n, k=k, h=h, seed=sample_seed,
                            p_in=0.3, p_out=0.05,
                            theta_distribution='power_law',
                            degree_correction_scale=1.5
                        )
                    elif gen_name == 'hb_lfr':
                        G, communities = hb_lfr(
                            n=n, tau1=2.5, h=h, seed=sample_seed,
                            mu=0.3,  # Fixed mixing parameter
                            max_iters=5000
                        )
                    else:
                        raise ValueError(f"Unknown generator: {gen_name}")

                    # Measure hub-bridging ratio
                    rho = compute_hub_bridging_ratio(G, communities)
                    gen_results['rho_samples'][h].append(rho)

                    # Measure modularity (using ground-truth communities)
                    Q = nx.community.modularity(G, communities)
                    gen_results['Q_samples'][h].append(Q)

                    # Store degree sequence for validation
                    degrees = sorted([d for n_id, d in G.degree()], reverse=True)
                    gen_results['degree_sequences'][h].append(degrees)

                    if (sample + 1) % 10 == 0:
                        mean_rho = np.mean(gen_results['rho_samples'][h])
                        mean_Q = np.mean(gen_results['Q_samples'][h])
                        logger.info(f"  Sample {sample+1}/{n_samples}: "
                                   f"ρ={rho:.3f} (μ={mean_rho:.3f}), "
                                   f"Q={Q:.3f} (μ={mean_Q:.3f})")

                except Exception as e:
                    logger.warning(f"  Sample {sample+1} failed: {e}")

        # Statistical analysis
        logger.info(f"\n{'=' * 70}")
        logger.info(f"STATISTICAL ANALYSIS: {gen_name.upper()}")
        logger.info(f"{'=' * 70}")

        # 1. Summary statistics by h
        logger.info("\nSummary by h value:")
        logger.info(f"{'h':>6} | {'ρ_HB':>10} | {'Q':>10}")
        logger.info("-" * 35)

        rho_means = {}
        Q_means = {}
        Q_stds = {}

        for h in h_values:
            rho_mean = np.mean(gen_results['rho_samples'][h])
            Q_mean = np.mean(gen_results['Q_samples'][h])
            Q_std = np.std(gen_results['Q_samples'][h])

            rho_means[h] = rho_mean
            Q_means[h] = Q_mean
            Q_stds[h] = Q_std

            logger.info(f"{h:6.2f} | {rho_mean:10.4f} | {Q_mean:10.4f} ± {Q_std:.4f}")

        # 2. Test correlation between ρ_HB and Q
        logger.info("\n--- Correlation Tests ---")

        # Flatten data for correlation
        all_rho = [rho for h in h_values for rho in gen_results['rho_samples'][h]]
        all_Q = [Q for h in h_values for Q in gen_results['Q_samples'][h]]

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(all_rho, all_Q)
        logger.info(f"Pearson correlation:  r = {pearson_r:+.4f}, p = {pearson_p:.4e}")

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(all_rho, all_Q)
        logger.info(f"Spearman correlation: r = {spearman_r:+.4f}, p = {spearman_p:.4e}")

        # Interpretation
        if abs(pearson_r) < 0.2:
            logger.info("✓ Weak correlation: Q appears independent of ρ_HB")
        elif abs(pearson_r) < 0.5:
            logger.info("⚠ Moderate correlation: Some dependency detected")
        else:
            logger.info("✗ Strong correlation: Significant dependency")

        # 3. ANOVA: Test if Q varies across h values
        logger.info("\n--- ANOVA Test ---")
        logger.info("H₀: Q is independent of h (no systematic variation)")

        Q_samples_by_h = [gen_results['Q_samples'][h] for h in h_values]
        f_stat, p_anova = f_oneway(*Q_samples_by_h)

        logger.info(f"F-statistic: F = {f_stat:.4f}")
        logger.info(f"p-value:     p = {p_anova:.4e}")

        # Calculate effect size regardless
        Q_range = max(Q_means.values()) - min(Q_means.values())
        Q_mean_overall = np.mean(all_Q)
        relative_range = Q_range / Q_mean_overall if Q_mean_overall != 0 else 0

        if p_anova > 0.05:
            logger.info("✓ Cannot reject H₀: Q independent of h")
        else:
            logger.info(f"⚠ Reject H₀: Q varies with h (p = {p_anova:.4e})")
            logger.info(f"  Effect size: ΔQ = {Q_range:.4f} ({relative_range:.1%} of mean)")

            if relative_range < 0.05:
                logger.info("  ✓ Effect size small (<5%), practically independent")

        # 4. Coefficient of variation in Q across h values
        logger.info("\n--- Variance Analysis ---")

        Q_cv_by_h = {h: Q_stds[h] / Q_means[h] if Q_means[h] != 0 else 0 for h in h_values}
        mean_cv = np.mean(list(Q_cv_by_h.values()))

        logger.info(f"Mean CV across h values: {mean_cv:.2%}")
        logger.info("Per-h CV:")
        for h in h_values:
            logger.info(f"  h={h:.2f}: CV={Q_cv_by_h[h]:.2%}")

        # 5. Degree sequence preservation check
        logger.info("\n--- Degree Preservation Check ---")
        logger.info("(Validating fixed constraint assumption)")

        # Compare degree sequences across h values
        # Use first sample from each h as reference
        ref_degrees = gen_results['degree_sequences'][h_values[0]][0]

        degree_preserved = True
        for h in h_values[1:]:
            if gen_results['degree_sequences'][h]:
                test_degrees = gen_results['degree_sequences'][h][0]

                # Check if degree distributions are similar
                mean_diff = abs(np.mean(ref_degrees) - np.mean(test_degrees))
                ref_mean = np.mean(ref_degrees)
                rel_diff = mean_diff / ref_mean if ref_mean != 0 else 0

                if rel_diff > 0.1:  # >10% difference
                    logger.warning(f"  h={h:.2f}: Mean degree differs by {rel_diff:.1%}")
                    degree_preserved = False
                else:
                    logger.info(f"  h={h:.2f}: Mean degree preserved ({rel_diff:.1%} diff)")

        if degree_preserved:
            logger.info("✓ Degree distributions preserved across h")
        else:
            logger.warning("⚠ Degree distributions vary across h")

        # Assessment
        logger.info(f"\n{'=' * 70}")
        logger.info("ASSESSMENT")
        logger.info(f"{'=' * 70}")

        # Pass criteria:
        # 1. |Pearson r| < 0.3 (weak correlation)
        # 2. ANOVA p > 0.05 OR effect size < 5%

        passes_correlation = abs(pearson_r) < 0.3

        if p_anova > 0.05:
            passes_anova = True
            effect_size_small = True
        else:
            effect_size_small = relative_range < 0.05
            passes_anova = effect_size_small

        passes_overall = passes_correlation and passes_anova

        logger.info(f"1. Correlation test:     {'✓ PASS' if passes_correlation else '✗ FAIL'}")
        logger.info(f"   |r| = {abs(pearson_r):.3f} {'<' if passes_correlation else '>'} 0.3")
        logger.info(f"2. Independence test:    {'✓ PASS' if passes_anova else '✗ FAIL'}")
        if p_anova > 0.05:
            logger.info(f"   ANOVA p = {p_anova:.3f} > 0.05")
        else:
            logger.info(f"   Effect size = {relative_range:.1%} {'<' if effect_size_small else '>'} 5%")
        logger.info(f"\nOverall:                 {'✓ PASS' if passes_overall else '✗ FAIL'}")
        logger.info(f"{'=' * 70}")

        # Store results
        gen_results['statistics'] = {
            'rho_means': rho_means,
            'Q_means': Q_means,
            'Q_stds': Q_stds,
            'Q_cv_by_h': Q_cv_by_h,
            'mean_cv': mean_cv,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'anova_f': f_stat,
            'anova_p': p_anova,
            'Q_range': Q_range,
            'relative_range': relative_range,
            'degree_preserved': degree_preserved
        }

        gen_results['assessment'] = {
            'passes_correlation': passes_correlation,
            'passes_anova': passes_anova,
            'effect_size_small': effect_size_small,
            'passes_overall': passes_overall
        }

        results[gen_name] = gen_results

    # Comparative summary if multiple generators
    if len(generators) > 1:
        logger.info("\n" + "=" * 70)
        logger.info("COMPARATIVE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"\n{'Generator':<12} | {'|r|':>8} | {'ANOVA p':>10} | {'Q range':>10} | {'Status':>10}")
        logger.info("-" * 60)

        for gen_name in generators:
            if gen_name in results:
                s = results[gen_name]['statistics']
                a = results[gen_name]['assessment']
                status = '✓ PASS' if a['passes_overall'] else '✗ FAIL'
                logger.info(f"{gen_name.upper():<12} | {abs(s['pearson_r']):>8.4f} | "
                           f"{s['anova_p']:>10.4f} | {s['Q_range']:>10.4f} | {status:>10}")

    return results
