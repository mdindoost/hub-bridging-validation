"""
Calibration Module
==================

This module provides utilities for calibrating the relationship
between the hub-bridging parameter h and the resulting hub-bridging
ratio rho_HB.

Since the relationship between h and rho_HB depends on network
parameters (n, tau1, mu, etc.), calibration is necessary to
achieve target rho_HB values.

References
----------
.. [1] Your PhD thesis or publication on hub-bridging benchmarks
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate, optimize

logger = logging.getLogger(__name__)


def calibrate_h_to_rho(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    h_values: Optional[List[float]] = None,
    h_range: Optional[Tuple[float, float]] = None,
    n_points: int = 20,
    n_samples: int = 10,
    seed: Optional[int] = None,
    target_rho: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Calibrate the h -> rho_HB relationship for a generator.

    This function generates multiple networks for each h value and
    computes the mean and standard deviation of rho_HB, allowing
    the construction of a calibration curve.

    Parameters
    ----------
    generator_func : Callable
        Generator function that takes parameters including 'h' and 'seed'
        and returns (graph, communities)
    generator_params : Dict[str, Any]
        Base parameters for the generator (excluding 'h')
    h_values : List[float], optional
        Explicit values of h to test. If None, uses h_range and n_points.
    h_range : Tuple[float, float], optional
        Range of h values to search (default: (0, 3))
    n_points : int, optional
        Number of h values to test when using h_range (default: 20)
    n_samples : int, optional
        Number of samples per h value (default: 10)
    seed : int, optional
        Base random seed for reproducibility
    target_rho : float, optional
        If provided, find h value that achieves this target ρ_HB
    verbose : bool, optional
        Print progress information (default: True)

    Returns
    -------
    Dict[str, Any]
        Calibration results with keys:
        - 'h_values': tested h values
        - 'rho_mean': mean rho_HB for each h
        - 'rho_std': standard deviation for each h
        - 'rho_samples': all samples (n_h x n_samples array)
        - 'interpolator': scipy interpolation function h -> rho
        - 'h_fitted': (if target_rho provided) h value for target

    Examples
    --------
    >>> from src.generators.hb_sbm import hb_sbm
    >>> params = {'n': 250, 'k': 5}
    >>> calibration = calibrate_h_to_rho(hb_sbm, params, n_samples=5, seed=42)
    >>> calibration['rho_mean'][0] < calibration['rho_mean'][-1]  # h=0 < h=1
    True
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    # Determine h values to test
    if h_values is None:
        if h_range is None:
            h_range = (0, 3)
        h_values = list(np.linspace(h_range[0], h_range[1], n_points))

    rng = np.random.default_rng(seed)

    if verbose:
        print(f"Calibrating h to achieve ρ_HB" +
              (f" = {target_rho:.2f}" if target_rho else ""))
        print(f"Testing {len(h_values)} h values with {n_samples} samples each")

    logger.info(
        f"Starting calibration with {len(h_values)} h values, "
        f"{n_samples} samples each"
    )

    rho_samples = np.zeros((len(h_values), n_samples))

    for i, h in enumerate(h_values):
        logger.debug(f"Calibrating h={h}")

        for j in range(n_samples):
            # Generate unique seed for this sample
            sample_seed = int(rng.integers(0, 2**31))

            try:
                # Generate network
                params = generator_params.copy()
                params["h"] = h
                params["seed"] = sample_seed

                G, communities = generator_func(**params)

                # Compute rho_HB
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples[i, j] = rho

            except Exception as e:
                logger.warning(f"Sample h={h}, j={j} failed: {e}")
                rho_samples[i, j] = np.nan

        mean_rho = np.nanmean(rho_samples[i])
        std_rho = np.nanstd(rho_samples[i])

        if verbose:
            print(f"  h={h:.2f}: ρ={mean_rho:.3f} ± {std_rho:.3f}")

        logger.debug(
            f"h={h}: rho_HB = {mean_rho:.4f} +/- {std_rho:.4f}"
        )

    # Compute statistics
    rho_mean = np.nanmean(rho_samples, axis=1)
    rho_std = np.nanstd(rho_samples, axis=1)

    # Create interpolator (h -> rho)
    valid_mask = ~np.isnan(rho_mean)
    if valid_mask.sum() >= 2:
        interpolator = interpolate.interp1d(
            np.array(h_values)[valid_mask],
            rho_mean[valid_mask],
            kind="cubic" if valid_mask.sum() >= 4 else "linear",
            fill_value="extrapolate",
        )
    else:
        interpolator = None
        logger.warning("Not enough valid samples for interpolation")

    result = {
        "h_values": np.array(h_values),
        "rho_mean": rho_mean,
        "rho_std": rho_std,
        "rho_samples": rho_samples,
        "interpolator": interpolator,
        "generator_params": generator_params,
        "calibration_curve": {
            "h_values": h_values,
            "rho_means": list(rho_mean),
            "rho_stds": list(rho_std),
        }
    }

    # Find h for target_rho if requested
    if target_rho is not None and interpolator is not None:
        # Create inverse interpolator (rho -> h)
        try:
            rho_to_h = interpolate.interp1d(
                rho_mean[valid_mask],
                np.array(h_values)[valid_mask],
                kind="cubic" if valid_mask.sum() >= 4 else "linear",
                fill_value="extrapolate",
            )
            h_fitted = float(rho_to_h(target_rho))
            h_fitted = np.clip(h_fitted, min(h_values), max(h_values))
            result["h_fitted"] = h_fitted
            result["target_rho"] = target_rho

            if verbose:
                print(f"\nFitted h = {h_fitted:.3f} for target ρ = {target_rho:.2f}")
        except Exception as e:
            logger.warning(f"Could not find h for target_rho: {e}")

    logger.info("Calibration complete")
    return result


def get_calibration_curve(
    calibration_result: Dict[str, Any],
    h_fine: Optional[NDArray[np.float64]] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Get a smooth calibration curve from calibration results.

    Parameters
    ----------
    calibration_result : Dict[str, Any]
        Result from calibrate_h_to_rho
    h_fine : NDArray[np.float64], optional
        Fine-grained h values for smooth curve.
        Default: 101 points from 0 to 1.

    Returns
    -------
    Tuple[NDArray, NDArray]
        (h_values, rho_values) for the smooth curve
    """
    if h_fine is None:
        h_fine = np.linspace(0, 1, 101)

    interpolator = calibration_result.get("interpolator")
    if interpolator is None:
        # Fall back to linear interpolation of measured points
        h_measured = calibration_result["h_values"]
        rho_measured = calibration_result["rho_mean"]
        rho_fine = np.interp(h_fine, h_measured, rho_measured)
    else:
        rho_fine = interpolator(h_fine)

    return h_fine, rho_fine


def h_for_target_rho(
    target_rho: float,
    calibration_result: Dict[str, Any],
    bounds: Tuple[float, float] = (0.0, 2.0),
) -> float:
    """
    Find h value that achieves a target rho_HB.

    Parameters
    ----------
    target_rho : float
        Target hub-bridging ratio
    calibration_result : Dict[str, Any]
        Result from calibrate_h_to_rho
    bounds : Tuple[float, float], optional
        Search bounds for h (default: (0.0, 2.0))

    Returns
    -------
    float
        Estimated h value to achieve target_rho

    Raises
    ------
    ValueError
        If target_rho is outside the calibrated range

    Examples
    --------
    >>> calibration = {'h_values': [0, 0.5, 1.0], 'rho_mean': [1.0, 1.5, 2.0]}
    >>> # This would need actual interpolator for real use
    """
    interpolator = calibration_result.get("interpolator")
    if interpolator is None:
        raise ValueError("Calibration result missing interpolator")

    # Check if target is in range
    rho_min = calibration_result["rho_mean"].min()
    rho_max = calibration_result["rho_mean"].max()

    if target_rho < rho_min or target_rho > rho_max:
        logger.warning(
            f"Target rho={target_rho} outside calibrated range "
            f"[{rho_min:.4f}, {rho_max:.4f}]"
        )

    # Create inverse function and find root
    def objective(h):
        return interpolator(h) - target_rho

    try:
        result = optimize.brentq(objective, bounds[0], bounds[1])
        return float(result)
    except ValueError:
        # Brentq failed, try minimize
        result = optimize.minimize_scalar(
            lambda h: (interpolator(h) - target_rho) ** 2,
            bounds=bounds,
            method="bounded",
        )
        return float(result.x)


def save_calibration(
    calibration_result: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Save calibration results to file.

    Parameters
    ----------
    calibration_result : Dict[str, Any]
        Result from calibrate_h_to_rho
    filepath : str
        Path to save file (.npz format)
    """
    # Extract serializable data (exclude interpolator)
    save_data = {
        "h_values": calibration_result["h_values"],
        "rho_mean": calibration_result["rho_mean"],
        "rho_std": calibration_result["rho_std"],
        "rho_samples": calibration_result["rho_samples"],
    }

    # Save generator params as string representation
    if "generator_params" in calibration_result:
        save_data["generator_params_str"] = str(calibration_result["generator_params"])

    np.savez(filepath, **save_data)
    logger.info(f"Saved calibration to {filepath}")


def load_calibration(filepath: str) -> Dict[str, Any]:
    """
    Load calibration results from file.

    Parameters
    ----------
    filepath : str
        Path to saved calibration file

    Returns
    -------
    Dict[str, Any]
        Calibration results (interpolator will be reconstructed)
    """
    data = np.load(filepath, allow_pickle=True)

    result = {
        "h_values": data["h_values"],
        "rho_mean": data["rho_mean"],
        "rho_std": data["rho_std"],
        "rho_samples": data["rho_samples"],
    }

    # Reconstruct interpolator
    valid_mask = ~np.isnan(result["rho_mean"])
    if valid_mask.sum() >= 2:
        result["interpolator"] = interpolate.interp1d(
            result["h_values"][valid_mask],
            result["rho_mean"][valid_mask],
            kind="cubic" if valid_mask.sum() >= 4 else "linear",
            fill_value="extrapolate",
        )
    else:
        result["interpolator"] = None

    logger.info(f"Loaded calibration from {filepath}")
    return result


def quick_calibration_check(
    generator_func: Callable,
    generator_params: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, float]:
    """
    Quick check of calibration at a few h values.

    Useful for verifying that a generator is working correctly
    before running full calibration.

    Parameters
    ----------
    generator_func : Callable
        Generator function
    generator_params : Dict[str, Any]
        Generator parameters (excluding 'h')
    seed : int, optional
        Random seed

    Returns
    -------
    Dict[str, float]
        Mapping from h value to measured rho_HB
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio

    h_values = [0.0, 0.5, 1.0]
    results = {}

    for h in h_values:
        params = generator_params.copy()
        params["h"] = h
        params["seed"] = seed

        try:
            G, communities = generator_func(**params)
            rho = compute_hub_bridging_ratio(G, communities)
            results[h] = rho
            logger.info(f"h={h}: rho_HB={rho:.4f}")
        except Exception as e:
            logger.error(f"h={h} failed: {e}")
            results[h] = np.nan

    return results


def extract_lfr_params_from_real(
    G: nx.Graph,
    communities: Union[List[set], Dict[int, int]],
) -> Dict[str, Any]:
    """
    Extract ESSENTIAL LFR parameters from a real network.

    Returns only parameters needed for structural matching:
    - n: network size
    - tau1: degree distribution exponent (power-law shape)
    - tau2: community size distribution exponent (power-law shape)
    - mu: mixing parameter (fraction of inter-community edges)

    IMPORTANT: We intentionally do NOT extract:
    - average_degree, max_degree, min_degree
    - min_community, max_community

    Why? These over-constrain the LFR generator and prevent the
    rewiring algorithm from achieving target hub-bridging ratios.
    The structural SHAPE (power-law exponents) matters more than
    exact values. Other properties emerge naturally from tau1, tau2.

    Parameters
    ----------
    G : networkx.Graph
        Real network
    communities : list of sets or dict
        Community structure (list of sets or node->community dict)

    Returns
    -------
    dict
        {
            'n': int,
            'tau1': float (degree distribution exponent),
            'tau2': float (community size distribution exponent),
            'mu': float (mixing parameter),
        }

    Notes
    -----
    - tau1: Fit power law to degree distribution
    - tau2: Fit power law to community sizes
    - mu: Compute actual mixing parameter from edges
    """
    from collections import Counter

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees) if degrees else 1
    min_degree = max(1, min(degrees)) if degrees else 1

    # Fit power law to degrees
    tau1 = 2.5  # Default
    try:
        import powerlaw
        fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
        tau1 = fit.power_law.alpha
        # Clip to reasonable range
        tau1 = np.clip(tau1, 2.0, 4.0)
    except Exception as e:
        logger.warning(f"Power law fit failed for degrees: {e}, using tau1={tau1}")

    # Parse communities
    if isinstance(communities, dict):
        # Convert dict to list of sets
        comm_dict: Dict[int, set] = {}
        for node, comm_id in communities.items():
            if comm_id not in comm_dict:
                comm_dict[comm_id] = set()
            comm_dict[comm_id].add(node)
        community_list = list(comm_dict.values())
    else:
        community_list = communities

    # Community sizes
    comm_sizes = [len(c) for c in community_list]

    min_community = min(comm_sizes) if comm_sizes else 10
    max_community = max(comm_sizes) if comm_sizes else n

    # Fit power law to community sizes
    tau2 = 1.5  # Default
    if len(comm_sizes) > 1:
        try:
            import powerlaw
            fit = powerlaw.Fit(comm_sizes, discrete=True, verbose=False)
            tau2 = fit.power_law.alpha
            # Clip to reasonable range
            tau2 = np.clip(tau2, 1.0, 3.0)
        except Exception as e:
            logger.warning(f"Power law fit failed for community sizes: {e}, using tau2={tau2}")

    # Compute mixing parameter mu
    # First, create node_to_community mapping
    node_to_comm: Dict[int, int] = {}
    for comm_id, members in enumerate(community_list):
        for node in members:
            node_to_comm[node] = comm_id

    inter_edges = 0
    for u, v in G.edges():
        u_comm = node_to_comm.get(u, -1)
        v_comm = node_to_comm.get(v, -2)
        if u_comm != v_comm:
            inter_edges += 1

    mu = inter_edges / m if m > 0 else 0.1
    # Clip to valid range for LFR
    mu = np.clip(mu, 0.01, 0.9)

    # ESSENTIAL PARAMETERS ONLY
    # We intentionally exclude avg_degree, min/max_community etc.
    # to avoid over-constraining the LFR generator
    params = {
        'n': n,
        'tau1': float(tau1),
        'tau2': float(tau2),
        'mu': float(mu),
    }

    logger.info(f"Extracted LFR params (essential only): n={n}, tau1={tau1:.2f}, tau2={tau2:.2f}, mu={mu:.3f}")

    return params


def extract_lfr_params_robust(
    G: nx.Graph,
    communities: Union[List[set], Dict[int, int]],
    target_rho: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Extract LFR parameters with adaptive bounds to ensure generation feasibility.

    Applies minimal intervention principle:
    - tau1 ≤ 2.8 (ensures sufficient hubs for hub-bridging)
    - tau2 ≤ 2.0 (LFR generator stability requirement)
    - mu adjusted proportionally to target_rho for high targets

    Parameters
    ----------
    G : networkx.Graph
        Real network
    communities : list of sets or dict
        Community structure (list of sets or node->community dict)
    target_rho : float, optional
        Target hub-bridging ratio (for adaptive mu adjustment)

    Returns
    -------
    dict
        LFR parameters with keys:
        - n, tau1, tau2, mu (adjusted values for generation)
        - tau1_raw, tau2_raw, mu_raw (original extracted values)
        - adjustments (dict describing applied changes)

    Notes
    -----
    This function preserves raw parameters when they work, and only
    applies minimal adjustments when necessary for generation stability
    or to achieve the target hub-bridging ratio.
    """
    # First, extract raw parameters using existing function
    raw_params = extract_lfr_params_from_real(G, communities)

    n = raw_params['n']
    tau1_raw = raw_params['tau1']
    tau2_raw = raw_params['tau2']
    mu_raw = raw_params['mu']

    adjustments = {}

    # 1. Cap tau1 at 2.8 (ensures hub availability for hub-bridging)
    # High tau1 means steep degree distribution with fewer hubs
    tau1 = min(tau1_raw, 2.8)
    if tau1 < tau1_raw:
        adjustments['tau1'] = f'capped from {tau1_raw:.2f} to {tau1:.2f} (ensures hub availability)'
        logger.info(f"tau1 capped: {tau1_raw:.2f} → {tau1:.2f}")

    # 2. Cap tau2 at 2.0 (LFR generation stability)
    # High tau2 causes LFR to fail generating valid community sizes
    tau2 = min(tau2_raw, 2.0)
    if tau2 < tau2_raw:
        adjustments['tau2'] = f'capped from {tau2_raw:.2f} to {tau2:.2f} (LFR stability)'
        logger.info(f"tau2 capped: {tau2_raw:.2f} → {tau2:.2f}")

    # 3. Adaptive mu adjustment for high hub-bridging targets
    mu = mu_raw
    if target_rho is not None and target_rho > 1.5:
        # Proportional minimum based on target difficulty
        # Higher target rho needs more inter-community edges for rewiring
        mu_min = 0.20 + 0.05 * (target_rho - 1.5)
        mu_min = min(mu_min, 0.35)  # Cap to preserve community structure

        if mu_raw < mu_min:
            mu = mu_min
            adjustments['mu'] = f'boosted from {mu_raw:.3f} to {mu:.3f} (enables ρ_target={target_rho:.2f})'
            logger.info(f"mu boosted: {mu_raw:.3f} → {mu:.3f} for target ρ={target_rho:.2f}")

    # Final safety bounds
    tau1 = float(np.clip(tau1, 2.0, 3.0))
    tau2 = float(np.clip(tau2, 1.1, 2.0))
    mu = float(np.clip(mu, 0.1, 0.6))

    params = {
        'n': n,
        'tau1': tau1,
        'tau2': tau2,
        'mu': mu,
        # Store raw values for reporting/transparency
        'tau1_raw': float(tau1_raw),
        'tau2_raw': float(tau2_raw),
        'mu_raw': float(mu_raw),
        # Track what adjustments were made
        'adjustments': adjustments,
    }

    if adjustments:
        logger.info(f"Applied {len(adjustments)} parameter adjustment(s): {list(adjustments.keys())}")
    else:
        logger.info("Using raw parameters (no adjustments needed)")

    return params


def extract_lfr_params_with_fallback(
    G: nx.Graph,
    communities: Union[List[set], Dict[int, int]],
    target_rho: Optional[float] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Wrapper with graceful degradation to canonical parameters.

    First tries adaptive bounds. If generation still fails,
    falls back to safe canonical parameters (tau1=2.5, tau2=1.5).

    Parameters
    ----------
    G : networkx.Graph
        Real network
    communities : list of sets or dict
        Community structure
    target_rho : float, optional
        Target hub-bridging ratio
    seed : int
        Random seed for validation test

    Returns
    -------
    dict
        LFR parameters (same format as extract_lfr_params_robust)
    """
    from .hb_lfr import hb_lfr

    # Try adaptive bounds first
    params = extract_lfr_params_robust(G, communities, target_rho)

    # Test if these parameters can generate a valid network
    try:
        G_test, _ = hb_lfr(
            n=params['n'],
            tau1=params['tau1'],
            tau2=params['tau2'],
            mu=params['mu'],
            h=0.0,
            max_iters=1000,
            seed=seed,
        )
        logger.info("✓ Extracted parameters validated successfully")
        return params

    except Exception as e:
        logger.warning(f"Parameter validation failed: {e}")
        logger.info("Falling back to canonical parameters (tau1=2.5, tau2=1.5)")

        # Canonical fallback - these always work
        params_canonical = {
            'n': params['n'],
            'tau1': 2.5,
            'tau2': 1.5,
            'mu': min(params['mu'], 0.45),  # Keep mu but cap it
            # Preserve raw values for reporting
            'tau1_raw': params['tau1_raw'],
            'tau2_raw': params['tau2_raw'],
            'mu_raw': params['mu_raw'],
            'adjustments': {
                'fallback': f'canonical parameters used after generation failed with tau1={params["tau1"]:.2f}, tau2={params["tau2"]:.2f}'
            },
        }

        return params_canonical


def fit_h_to_real_network(
    G_real: nx.Graph,
    communities_real: Union[List[set], Dict[int, int]],
    lfr_params: Optional[Dict[str, Any]] = None,
    n_calibration_samples: int = 10,
    n_validation_samples: int = 5,
    h_range: Tuple[float, float] = (0.0, 2.5),
    n_h_points: int = 21,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Find h value that produces rho_HB closest to real network.

    Algorithm:
    1. Compute target rho_HB from real network
    2. Generate calibration curve: sample networks at different h values
    3. Interpolate to find h that gives target rho_HB
    4. Validate with additional samples

    Parameters
    ----------
    G_real : networkx.Graph
        Real network
    communities_real : list of sets or dict
        Real community structure
    lfr_params : dict, optional
        LFR parameters. If None, extracted from real network.
    n_calibration_samples : int
        Number of networks to generate per h value for calibration
    n_validation_samples : int
        Number of networks for final validation
    h_range : tuple
        (min_h, max_h) to test
    n_h_points : int
        Number of h values to test in calibration
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        {
            'h_fitted': float,
            'rho_target': float,
            'rho_achieved_mean': float,
            'rho_achieved_std': float,
            'calibration_curve': {
                'h_values': array,
                'rho_mean': array,
                'rho_std': array
            },
            'relative_error': float,
            'validation_samples': list of rho_HB values
        }
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio
    from .hb_lfr import hb_lfr

    rng = np.random.default_rng(seed)

    # Step 1: Compute target rho_HB from real network
    rho_target = compute_hub_bridging_ratio(G_real, communities_real)
    logger.info(f"Target rho_HB = {rho_target:.3f}")

    # Extract LFR params if not provided (using robust extraction with adaptive bounds)
    if lfr_params is None:
        lfr_params = extract_lfr_params_robust(G_real, communities_real, target_rho=rho_target)
        # Log any adjustments made
        if lfr_params.get('adjustments'):
            logger.info(f"Parameter adjustments applied: {list(lfr_params['adjustments'].keys())}")

    # Step 2: Generate calibration curve
    h_values = np.linspace(h_range[0], h_range[1], n_h_points)
    rho_means = []
    rho_stds = []

    logger.info(f"Generating calibration curve ({n_h_points} points, {n_calibration_samples} samples each)")

    for h in h_values:
        rho_samples = []

        for sample in range(n_calibration_samples):
            sample_seed = int(rng.integers(0, 2**31))

            try:
                G, communities = hb_lfr(
                    n=lfr_params['n'],
                    tau1=lfr_params['tau1'],
                    tau2=lfr_params.get('tau2', 1.5),
                    mu=lfr_params['mu'],
                    average_degree=lfr_params.get('average_degree'),
                    min_community=lfr_params.get('min_community'),
                    max_community=lfr_params.get('max_community'),
                    h=h,
                    seed=sample_seed,
                    # BUG 2 FIX: Pass real target to hb_lfr
                    target_rho=rho_target,
                )
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples.append(rho)
            except Exception as e:
                logger.debug(f"Failed at h={h}, sample {sample}: {e}")
                continue

        if len(rho_samples) > 0:
            rho_means.append(np.mean(rho_samples))
            rho_stds.append(np.std(rho_samples))
        else:
            logger.warning(f"All samples failed at h={h}")
            rho_means.append(np.nan)
            rho_stds.append(np.nan)

    rho_means = np.array(rho_means)
    rho_stds = np.array(rho_stds)

    # Remove NaN values for interpolation
    valid_idx = ~np.isnan(rho_means)
    h_valid = h_values[valid_idx]
    rho_valid = rho_means[valid_idx]

    if len(h_valid) < 3:
        raise ValueError("Not enough valid calibration points")

    # Step 3: Interpolate to find h for target rho
    try:
        rho_to_h = interpolate.interp1d(
            rho_valid, h_valid,
            kind='cubic' if len(h_valid) >= 4 else 'linear',
            fill_value='extrapolate',
            bounds_error=False
        )
        h_fitted = float(rho_to_h(rho_target))
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}, using nearest point")
        # Fallback: find closest rho
        closest_idx = np.argmin(np.abs(rho_valid - rho_target))
        h_fitted = h_valid[closest_idx]

    # Clip to valid range
    h_fitted = np.clip(h_fitted, h_range[0], h_range[1])

    logger.info(f"Fitted h = {h_fitted:.3f}")

    # Step 4: Validate
    validation_rho = []

    for sample in range(n_validation_samples):
        sample_seed = int(rng.integers(0, 2**31))

        try:
            G, communities = hb_lfr(
                n=lfr_params['n'],
                tau1=lfr_params['tau1'],
                tau2=lfr_params.get('tau2', 1.5),
                mu=lfr_params['mu'],
                average_degree=lfr_params.get('average_degree'),
                min_community=lfr_params.get('min_community'),
                max_community=lfr_params.get('max_community'),
                h=h_fitted,
                seed=sample_seed,
                # BUG 2 FIX: Pass real target to hb_lfr
                target_rho=rho_target,
            )
            rho = compute_hub_bridging_ratio(G, communities)
            validation_rho.append(rho)
        except Exception as e:
            logger.warning(f"Validation sample {sample} failed: {e}")

    if len(validation_rho) == 0:
        raise ValueError("All validation samples failed")

    achieved_mean = np.mean(validation_rho)
    achieved_std = np.std(validation_rho)
    relative_error = abs(achieved_mean - rho_target) / rho_target if rho_target != 0 else 0

    logger.info(f"Validation: rho_HB = {achieved_mean:.3f} +/- {achieved_std:.3f} (error: {relative_error:.1%})")

    return {
        'h_fitted': h_fitted,
        'rho_target': rho_target,
        'rho_achieved_mean': achieved_mean,
        'rho_achieved_std': achieved_std,
        'calibration_curve': {
            'h_values': h_values.tolist(),
            'rho_mean': rho_means.tolist(),
            'rho_std': rho_stds.tolist()
        },
        'relative_error': relative_error,
        'validation_samples': validation_rho,
        'lfr_params': lfr_params,
    }


def fit_h_to_real_network_extended(
    G_real: nx.Graph,
    communities_real: Union[List[set], Dict[int, int]],
    lfr_params: Optional[Dict[str, Any]] = None,
    n_calibration_samples: int = 5,  # Reduced from 10 for speed
    n_validation_samples: int = 3,   # Reduced from 5 for speed
    h_range: Tuple[float, float] = (-0.5, 3.5),
    n_h_points: int = 25,  # Reduced from 31
    adaptive: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extended h fitting for extreme hub-bridging and hub-isolation networks.

    NEW FEATURES:
    - Extended h_range to handle hub-isolation (h < 0) and extreme hub-bridging (h > 3)
    - Adaptive range adjustment based on target ρ_HB
    - Better error handling for extreme cases
    - Reports partial success if perfect match impossible

    Algorithm:
    1. Compute target ρ_HB from real network
    2. Adjust h_range if target is extreme (adaptive)
    3. Generate calibration curve
    4. Interpolate to find h
    5. Validate
    6. Report achievability

    Parameters
    ----------
    G_real : networkx.Graph
        Real network
    communities_real : list of sets or dict
        Real community structure
    lfr_params : dict, optional
        LFR parameters. If None, extracted from real network.
    n_calibration_samples : int
        Number of networks per h value for calibration
    n_validation_samples : int
        Number of networks for final validation
    h_range : tuple
        (min_h, max_h) to test - extended to handle extreme cases
    n_h_points : int
        Number of h values to test
    adaptive : bool
        Whether to adapt h_range based on target ρ_HB
    seed : int, optional
        Random seed

    Returns
    -------
    dict
        {
            'h_fitted': float,
            'rho_target': float,
            'rho_target_fitted': float (may differ if target outside achievable range),
            'rho_achieved_mean': float,
            'rho_achieved_std': float,
            'calibration_curve': dict,
            'relative_error': float,
            'achievable': bool (whether target was in achievable range),
            'achievable_range': tuple (min, max achievable ρ_HB),
            'validation_samples': list,
            'lfr_params': dict
        }
    """
    from ..metrics.hub_bridging import compute_hub_bridging_ratio
    from .hb_lfr import hb_lfr

    rng = np.random.default_rng(seed)

    # Step 1: Compute target rho_HB from real network
    rho_target = compute_hub_bridging_ratio(G_real, communities_real)
    logger.info(f"Target ρ_HB = {rho_target:.3f}")

    # Extract LFR params if not provided (using robust extraction with adaptive bounds)
    if lfr_params is None:
        lfr_params = extract_lfr_params_robust(G_real, communities_real, target_rho=rho_target)
        # Log any adjustments made
        if lfr_params.get('adjustments'):
            logger.info(f"Parameter adjustments applied: {list(lfr_params['adjustments'].keys())}")

    # Step 2: Adaptive range adjustment
    h_min, h_max = h_range
    adjusted_n_h_points = n_h_points

    if adaptive:
        if rho_target > 6.0:
            logger.info(f"Extreme hub-bridging detected (ρ={rho_target:.2f}), extending h_range")
            h_max = min(5.0, h_max + 1.5)
            adjusted_n_h_points = 41
        elif rho_target > 4.0:
            logger.info(f"Strong hub-bridging detected (ρ={rho_target:.2f}), extending h_range")
            h_max = min(4.0, h_max + 0.5)
        elif rho_target < 0.6:
            logger.info(f"Extreme hub-isolation detected (ρ={rho_target:.2f}), extending negative h_range")
            h_min = max(-1.5, h_min - 1.0)
            adjusted_n_h_points = 41
        elif rho_target < 0.8:
            logger.info(f"Hub-isolation detected (ρ={rho_target:.2f}), extending negative h_range")
            h_min = max(-1.0, h_min - 0.5)

    # Step 3: SMART CALIBRATION WITH RANGE DETECTION
    #
    # 1. First probe to find achievable ρ range
    # 2. If target outside range → skip search, use best boundary
    # 3. If target inside range → binary search to find exact h

    def generate_samples(h_val, n_samples=2, use_target_rho=True):
        """Generate n networks and return mean rho_HB.

        Args:
            h_val: Hub-bridging parameter to test
            n_samples: Number of networks to generate
            use_target_rho: If True, pass real target to hb_lfr (Bug 2 fix)
        """
        rho_list = []
        for _ in range(n_samples):
            sample_seed = int(rng.integers(0, 2**31))
            try:
                G, communities = hb_lfr(
                    n=lfr_params['n'],
                    tau1=lfr_params['tau1'],
                    tau2=lfr_params.get('tau2', 1.5),
                    mu=lfr_params['mu'],
                    average_degree=lfr_params.get('average_degree'),
                    min_community=lfr_params.get('min_community'),
                    max_community=lfr_params.get('max_community'),
                    h=h_val,
                    seed=sample_seed,
                    max_iters=1500,
                    # BUG 2 FIX: Pass real target to hb_lfr instead of letting
                    # it compute formula-based target from h
                    target_rho=rho_target if use_target_rho else None,
                )
                rho = compute_hub_bridging_ratio(G, communities)
                rho_list.append(rho)
            except Exception as e:
                logger.debug(f"Failed at h={h_val:.2f}: {e}")
                continue
        if len(rho_list) == 0:
            return None, None
        return np.mean(rho_list), np.std(rho_list) if len(rho_list) > 1 else 0.0

    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"CALIBRATION: Finding h for target ρ = {rho_target:.3f}")
    logger.info(f"=" * 60)

    # Track all tested points for calibration curve
    tested_h = []
    tested_rho = []
    tested_std = []

    # =========================================================
    # PHASE 1: Probe to find achievable range
    # =========================================================
    logger.info(f"")
    logger.info(f"PHASE 1: Probing achievable ρ range...")
    logger.info(f"-" * 40)

    # Test lower bound (h_min)
    rho_lower, _ = generate_samples(h_min, n_samples=1)
    if rho_lower is not None:
        tested_h.append(h_min)
        tested_rho.append(rho_lower)
        tested_std.append(0.0)
        logger.info(f"  Lower bound:  h={h_min:.2f} → ρ={rho_lower:.3f}")
    else:
        logger.warning(f"  Lower bound:  h={h_min:.2f} → FAILED")
        rho_lower = 0.0

    # Test middle point (h=0, baseline) FIRST - needed for upper bound estimation
    rho_middle, _ = generate_samples(0.0, n_samples=1)
    if rho_middle is not None:
        tested_h.append(0.0)
        tested_rho.append(rho_middle)
        tested_std.append(0.0)
        logger.info(f"  Middle (h=0): h=0.00 → ρ={rho_middle:.3f}")
    else:
        logger.warning(f"  Middle (h=0): h=0.00 → FAILED")
        rho_middle = None

    # Test upper bound (h_max)
    rho_upper, _ = generate_samples(h_max, n_samples=1)
    upper_probe_failed = False
    if rho_upper is not None:
        tested_h.append(h_max)
        tested_rho.append(rho_upper)
        tested_std.append(0.0)
        logger.info(f"  Upper bound:  h={h_max:.2f} → ρ={rho_upper:.3f}")
    else:
        upper_probe_failed = True
        # BUG 1 FIX: Estimate rho_upper based on h=0 result and mu
        # Instead of arbitrary 10.0, use conservative estimate
        mu = lfr_params.get('mu', 0.3)
        if rho_middle is not None:
            # Factor depends on mixing parameter - lower mu means lower ceiling
            if mu < 0.2:
                factor = 1.3  # Low mixing → lower achievable ceiling
            elif mu < 0.4:
                factor = 1.5  # Medium mixing
            else:
                factor = 2.0  # High mixing allows more rewiring headroom

            rho_upper = rho_middle * factor
            logger.warning(
                f"  Upper bound:  h={h_max:.2f} → FAILED (generation error)"
            )
            logger.warning(
                f"  Estimating rho_upper = {rho_upper:.3f} "
                f"(h=0 result × {factor:.1f} for mu={mu:.2f})"
            )
        else:
            # Both middle and upper failed - very problematic network
            rho_upper = 2.0  # Conservative default
            logger.error(
                f"  Upper bound:  h={h_max:.2f} → FAILED"
            )
            logger.error(
                f"  Both upper and middle probes failed! "
                f"Using conservative default rho_upper = {rho_upper:.1f}"
            )

    # Determine achievable range (use all successful probes)
    all_probes = [rho_lower]
    if not upper_probe_failed:
        all_probes.append(rho_upper)
    if rho_middle is not None:
        all_probes.append(rho_middle)

    rho_achievable_min = min(all_probes)
    # For max: use actual upper if available, else use estimated
    if not upper_probe_failed:
        rho_achievable_max = max(all_probes)
    else:
        # Use estimated upper bound (already computed above)
        rho_achievable_max = rho_upper
        logger.info(f"  Note: Upper bound is ESTIMATED (probe failed)")

    logger.info(f"")
    logger.info(f"  Achievable range: [{rho_achievable_min:.3f}, {rho_achievable_max:.3f}]")
    logger.info(f"  Target:           {rho_target:.3f}")

    # =========================================================
    # PHASE 2: Check if target is achievable
    # =========================================================
    logger.info(f"")
    logger.info(f"PHASE 2: Checking target achievability...")
    logger.info(f"-" * 40)

    target_achievable = rho_achievable_min <= rho_target <= rho_achievable_max
    skip_binary_search = False

    if rho_target > rho_achievable_max:
        # Target too high - use upper bound
        logger.warning(f"")
        logger.warning(f"  *** TARGET TOO HIGH ***")
        logger.warning(f"  Target ρ={rho_target:.3f} > max achievable ρ={rho_achievable_max:.3f}")
        logger.warning(f"  Using best available: h={h_max:.2f} → ρ={rho_upper:.3f}")
        logger.warning(f"")
        best_h = h_max
        best_rho = rho_upper
        best_diff = abs(rho_upper - rho_target)
        skip_binary_search = True

    elif rho_target < rho_achievable_min:
        # Target too low - use lower bound
        logger.warning(f"")
        logger.warning(f"  *** TARGET TOO LOW ***")
        logger.warning(f"  Target ρ={rho_target:.3f} < min achievable ρ={rho_achievable_min:.3f}")
        logger.warning(f"  Using best available: h={h_min:.2f} → ρ={rho_lower:.3f}")
        logger.warning(f"")
        best_h = h_min
        best_rho = rho_lower
        best_diff = abs(rho_lower - rho_target)
        skip_binary_search = True

    else:
        logger.info(f"  Target is ACHIEVABLE within range")
        logger.info(f"  Proceeding with binary search...")
        best_h = (h_min + h_max) / 2
        best_rho = None
        best_diff = float('inf')

    # =========================================================
    # PHASE 3: Binary search (only if target is achievable)
    # =========================================================
    if not skip_binary_search:
        logger.info(f"")
        logger.info(f"PHASE 3: Binary search for optimal h...")
        logger.info(f"-" * 40)

        search_low = h_min
        search_high = h_max
        max_iterations = 8
        tolerance = 0.15

        for iteration in range(max_iterations):
            if search_high - search_low < tolerance:
                logger.info(f"  Converged: range [{search_low:.2f}, {search_high:.2f}] < {tolerance}")
                break

            h_mid = (search_low + search_high) / 2
            rho_mid, std_mid = generate_samples(h_mid, n_samples=2)

            if rho_mid is None:
                h_mid = h_mid + 0.05
                rho_mid, std_mid = generate_samples(h_mid, n_samples=2)
                if rho_mid is None:
                    logger.warning(f"  h={h_mid:.2f} failed, narrowing range")
                    search_high = h_mid
                    continue

            tested_h.append(h_mid)
            tested_rho.append(rho_mid)
            tested_std.append(std_mid if std_mid else 0.0)

            diff = abs(rho_mid - rho_target)
            logger.info(f"  h={h_mid:.2f} → ρ={rho_mid:.3f} (diff={diff:.3f})")

            if diff < best_diff:
                best_diff = diff
                best_h = h_mid
                best_rho = rho_mid

            if diff < 0.05:
                logger.info(f"  *** EXCELLENT MATCH FOUND ***")
                break

            if rho_mid < rho_target:
                search_low = h_mid
            else:
                search_high = h_mid
    else:
        logger.info(f"")
        logger.info(f"PHASE 3: Binary search SKIPPED (target not achievable)")

    # Convert to arrays
    h_values = np.array(tested_h)
    rho_means = np.array(tested_rho)
    rho_stds = np.array(tested_std)

    # Sort by h value
    sort_idx = np.argsort(h_values)
    h_values = h_values[sort_idx]
    rho_means = rho_means[sort_idx]
    rho_stds = rho_stds[sort_idx]

    # Remove NaN values
    valid_idx = ~np.isnan(rho_means)
    h_valid = h_values[valid_idx]
    rho_valid = rho_means[valid_idx]
    rho_std_valid = rho_stds[valid_idx]

    if len(h_valid) < 1:
        raise ValueError(f"Calibration failed - no valid samples")

    # Step 4: Summary and final h selection
    rho_min = rho_valid.min()
    rho_max = rho_valid.max()
    achievable = rho_min <= rho_target <= rho_max

    # Use best_h found during calibration
    h_fitted = best_h
    rho_target_fitted = rho_target if achievable else (rho_max if rho_target > rho_max else rho_min)

    # Print summary
    logger.info(f"")
    logger.info(f"=" * 60)
    logger.info(f"CALIBRATION RESULT")
    logger.info(f"=" * 60)
    logger.info(f"  Target ρ:      {rho_target:.3f}")
    logger.info(f"  Achievable:    [{rho_min:.3f}, {rho_max:.3f}]")
    logger.info(f"  Best h:        {h_fitted:.3f}")
    logger.info(f"  Best ρ:        {best_rho:.3f}" if best_rho else f"  Best ρ:        N/A")
    logger.info(f"  Difference:    {best_diff:.3f}")
    if achievable:
        logger.info(f"  Status:        TARGET ACHIEVABLE")
    else:
        logger.warning(f"  Status:        TARGET NOT ACHIEVABLE")
    logger.info(f"=" * 60)
    logger.info(f"")

    # Step 5: Validate
    # Use reduced iterations if target not achievable (we already know the ceiling)
    validation_max_iters = 500 if not achievable else 5000
    if not achievable:
        logger.info(f"Using reduced max_iters={validation_max_iters} (target not achievable)")

    validation_rho = []

    for sample in range(n_validation_samples):
        sample_seed = int(rng.integers(0, 2**31))

        try:
            G, communities = hb_lfr(
                n=lfr_params['n'],
                tau1=lfr_params['tau1'],
                tau2=lfr_params.get('tau2', 1.5),
                mu=lfr_params['mu'],
                average_degree=lfr_params.get('average_degree'),
                min_community=lfr_params.get('min_community'),
                max_community=lfr_params.get('max_community'),
                h=h_fitted,
                seed=sample_seed,
                max_iters=validation_max_iters,
                # BUG 2 FIX: Pass real target to hb_lfr
                target_rho=rho_target,
            )
            rho = compute_hub_bridging_ratio(G, communities)
            validation_rho.append(rho)
        except Exception as e:
            logger.warning(f"Validation sample {sample} failed: {e}")

    if len(validation_rho) == 0:
        raise ValueError("All validation samples failed")

    achieved_mean = np.mean(validation_rho)
    achieved_std = np.std(validation_rho)
    relative_error = abs(achieved_mean - rho_target) / rho_target if rho_target != 0 else 0

    logger.info(f"Validation: ρ_HB = {achieved_mean:.3f} ± {achieved_std:.3f} "
                f"(error: {relative_error:.1%})")

    # Step 6: Compute improvement vs standard LFR (h=0) for non-achievable targets
    if not achievable:
        # Standard LFR typically produces ρ ≈ 0.8-1.2 depending on network
        standard_rho_estimate = 0.9
        improvement_vs_standard = 1 - abs(achieved_mean - rho_target) / abs(standard_rho_estimate - rho_target)
        logger.info(f"Partial match: {improvement_vs_standard*100:.1f}% improvement over standard LFR")
    else:
        improvement_vs_standard = 1.0

    return {
        'h_fitted': h_fitted,
        'rho_target': rho_target,
        'rho_target_fitted': rho_target_fitted,
        'rho_achieved_mean': achieved_mean,
        'rho_achieved_std': achieved_std,
        'calibration_curve': {
            'h_values': h_values.tolist(),
            'rho_mean': rho_means.tolist(),
            'rho_std': rho_stds.tolist()
        },
        'relative_error': relative_error,
        'achievable': achievable,
        'achievable_range': (float(rho_min), float(rho_max)),
        'improvement_vs_standard': improvement_vs_standard,
        'validation_samples': validation_rho,
        'lfr_params': lfr_params,
        'h_range_used': (h_min, h_max),
    }
