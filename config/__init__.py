"""Configuration module for Hub-Bridging Validation Framework."""

from pathlib import Path
import yaml
from typing import Any, Dict

CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_name : str
        Name of the configuration file (without .yaml extension)

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_validation_config() -> Dict[str, Any]:
    """Load the validation configuration."""
    return load_config("validation_config")


def load_network_params() -> Dict[str, Any]:
    """Load the network parameters configuration."""
    return load_config("network_params")


__all__ = [
    "load_config",
    "load_validation_config",
    "load_network_params",
    "CONFIG_DIR",
]
