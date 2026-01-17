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
