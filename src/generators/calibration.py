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
    Extract LFR parameters from a real network with community structure.

    This estimates parameters that would generate a network with similar
    properties to the input network.

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
            'tau1': float (degree exponent),
            'tau2': float (community size exponent),
            'mu': float (mixing parameter),
            'average_degree': float,
            'max_degree': int,
            'min_degree': int,
            'min_community': int,
            'max_community': int,
        }

    Notes
    -----
    - tau1: Fit power law to degree distribution using powerlaw package
    - tau2: Fit power law to community sizes
    - mu: Compute actual mixing parameter from edges
    - Handle edge cases (non-power-law networks, single community, etc.)
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

    params = {
        'n': n,
        'tau1': float(tau1),
        'tau2': float(tau2),
        'mu': float(mu),
        'average_degree': float(avg_degree),
        'max_degree': int(max_degree),
        'min_degree': int(min_degree),
        'min_community': int(min_community),
        'max_community': int(max_community),
    }

    logger.info(f"Extracted LFR params: n={n}, tau1={tau1:.2f}, tau2={tau2:.2f}, mu={mu:.3f}")

    return params


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

    # Extract LFR params if not provided
    if lfr_params is None:
        lfr_params = extract_lfr_params_from_real(G_real, communities_real)

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
    n_calibration_samples: int = 10,
    n_validation_samples: int = 5,
    h_range: Tuple[float, float] = (-0.5, 3.5),
    n_h_points: int = 31,
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

    # Extract LFR params if not provided
    if lfr_params is None:
        lfr_params = extract_lfr_params_from_real(G_real, communities_real)

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

    # Step 3: Generate calibration curve
    # SMART FILTERING: Skip h values that are clearly wrong direction
    h_values_full = np.linspace(h_min, h_max, adjusted_n_h_points)

    # Filter based on target ρ_HB
    if rho_target > 1.5:
        # Hub-bridging: skip negative h values (they produce ρ < 1)
        h_values = h_values_full[h_values_full >= -0.1]
        logger.info(f"Target ρ={rho_target:.2f} > 1.5: skipping h < -0.1")
    elif rho_target < 0.9:
        # Hub-isolation: skip high h values (they produce ρ > 1.5)
        h_values = h_values_full[h_values_full <= 1.5]
        logger.info(f"Target ρ={rho_target:.2f} < 0.9: skipping h > 1.5")
    else:
        # Neutral: use middle range
        h_values = h_values_full[(h_values_full >= -0.3) & (h_values_full <= 2.5)]
        logger.info(f"Target ρ={rho_target:.2f} near 1: using h ∈ [-0.3, 2.5]")

    rho_means = []
    rho_stds = []

    logger.info(f"Generating calibration curve ({len(h_values)} points, {n_calibration_samples} samples each)")

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
                    max_iters=2000,  # Reduced for calibration (faster)
                )
                rho = compute_hub_bridging_ratio(G, communities)
                rho_samples.append(rho)
            except Exception as e:
                logger.debug(f"Failed at h={h:.2f}, sample {sample}: {e}")
                continue

        # Need at least half of samples to succeed
        if len(rho_samples) >= max(1, n_calibration_samples // 2):
            rho_means.append(np.mean(rho_samples))
            rho_stds.append(np.std(rho_samples))
        else:
            logger.warning(f"Insufficient samples at h={h:.2f}, marking as invalid")
            rho_means.append(np.nan)
            rho_stds.append(np.nan)

    rho_means = np.array(rho_means)
    rho_stds = np.array(rho_stds)

    # Remove NaN values
    valid_idx = ~np.isnan(rho_means)
    h_valid = h_values[valid_idx]
    rho_valid = rho_means[valid_idx]
    rho_std_valid = rho_stds[valid_idx]

    if len(h_valid) < 3:
        raise ValueError(f"Not enough valid calibration points (only {len(h_valid)})")

    # Step 4: Check achievability and interpolate
    rho_min = rho_valid.min()
    rho_max = rho_valid.max()
    achievable = rho_min <= rho_target <= rho_max

    if not achievable:
        if rho_target < rho_min:
            logger.warning(f"Target ρ={rho_target:.3f} below achievable range "
                          f"[{rho_min:.3f}, {rho_max:.3f}]")
            logger.warning(f"Will fit to minimum achievable value with slight extrapolation")
            rho_target_fitted = rho_min * 1.05  # Slight extrapolation
        else:
            logger.warning(f"Target ρ={rho_target:.3f} above achievable range "
                          f"[{rho_min:.3f}, {rho_max:.3f}]")
            logger.warning(f"Will fit to maximum achievable value with slight extrapolation")
            rho_target_fitted = rho_max * 0.95  # Slight extrapolation
    else:
        rho_target_fitted = rho_target
        logger.info(f"Target is achievable within range [{rho_min:.3f}, {rho_max:.3f}]")

    # Create interpolation function (rho -> h)
    try:
        # Sort by rho for monotonic interpolation
        sort_idx = np.argsort(rho_valid)
        rho_sorted = rho_valid[sort_idx]
        h_sorted = h_valid[sort_idx]

        rho_to_h = interpolate.interp1d(
            rho_sorted, h_sorted,
            kind='cubic' if len(h_valid) >= 4 else 'linear',
            fill_value='extrapolate',
            bounds_error=False
        )
        h_fitted = float(rho_to_h(rho_target_fitted))
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}, using nearest point")
        closest_idx = np.argmin(np.abs(rho_valid - rho_target_fitted))
        h_fitted = h_valid[closest_idx]

    # Clip to extended valid range
    h_fitted = np.clip(h_fitted, h_min, h_max)
    logger.info(f"Fitted h = {h_fitted:.3f}")

    # Step 5: Validate
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
                max_iters=5000,
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
