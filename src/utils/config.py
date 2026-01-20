"""Configuration management module for loading and accessing YAML configuration files.

This module provides functions to load YAML configuration files, merge environment
variable overrides, and provide typed access to configuration values with validation.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Global cache for configuration
_config_cache: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file and merge with environment variable overrides.

    Args:
        config_path: Path to YAML configuration file. If None, defaults to
            `models/config.yaml` relative to project root.

    Returns:
        Dictionary containing configuration values with environment variable
        overrides applied.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        ValueError: If required configuration sections are missing.

    Example:
        >>> config = load_config("models/config.yaml")
        >>> model_name = config["model"]["name"]
    """
    if config_path is None:
        # Default to models/config.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = str(project_root / "models" / "config.yaml")

    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            f"Please create it from {config_path}.example"
        )

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration file: {e}") from e

    if config is None:
        config = {}

    # Merge environment variable overrides
    config = _merge_env_overrides(config)

    # Validate configuration
    _validate_config(config)

    return config


def get_config() -> Dict[str, Any]:
    """Get cached configuration or load default configuration.

    This function implements a singleton pattern, caching the configuration
    after the first load to avoid reloading the YAML file on every call.

    Returns:
        Dictionary containing configuration values.

    Example:
        >>> config = get_config()
        >>> batch_size = config["training"]["batch_size"]
    """
    global _config_cache

    if _config_cache is None:
        _config_cache = load_config()

    return _config_cache


def _merge_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge environment variable overrides into configuration dictionary.

    Environment variables use double underscore (__) convention for nested keys.
    For example, `DATA__TRAIN_SPLIT` maps to `config["data"]["train_split"]`.

    Args:
        config: Configuration dictionary to merge overrides into.

    Returns:
        Configuration dictionary with environment variable overrides applied.
    """
    merged = config.copy()

    for key, value in os.environ.items():
        if "__" in key:
            # Split on double underscore to get nested path
            parts = key.split("__")
            _set_nested_value(merged, parts, value)

    return merged


def _set_nested_value(config: Dict[str, Any], path: List[str], value: Any) -> None:
    """Set a nested value in configuration dictionary.

    Args:
        config: Configuration dictionary to modify.
        path: List of keys representing the nested path (e.g., ["data", "train_split"]).
        value: Value to set at the nested path.
    """
    current = config

    # Navigate to the parent of the target key
    for i, key in enumerate(path[:-1]):
        key_lower = key.lower()
        if key_lower not in current:
            current[key_lower] = {}
        elif not isinstance(current[key_lower], dict):
            # If the path conflicts with an existing non-dict value, log warning and skip
            conflict_path = ".".join(path[: i + 1])
            logger.warning(
                f"Cannot set environment variable override: '{'__'.join(path)}' "
                f"conflicts with existing non-dict value at path '{conflict_path}'"
            )
            return
        current = current[key_lower]

    # Set the final value, converting string to appropriate type
    final_key = path[-1].lower()
    converted_value = _convert_env_value(value)
    current[final_key] = converted_value


def _convert_env_value(value: str) -> Any:
    """Convert environment variable string to appropriate Python type.

    Args:
        value: String value from environment variable.

    Returns:
        Converted value (bool, int, float, or str).
    """
    # Try boolean
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try integer
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as string
    return value


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration dictionary to YAML file.

    Args:
        config: Configuration dictionary to save.
        config_path: Path to YAML file to save configuration to.

    Raises:
        IOError: If the file cannot be written.

    Example:
        >>> config = {"model": {"name": "test"}, "training": {"lr": 0.001}}
        >>> save_config(config, "models/config_test.yaml")
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, indent=2)
    except IOError as e:
        raise IOError(f"Failed to write configuration file: {e}") from e

    logger.info(f"Saved configuration to {config_path}")


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate that required configuration sections exist.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If required sections are missing.
    """
    required_sections = ["model", "training", "data", "paths"]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Required configuration section '{section}' is missing. "
                f"Please check your configuration file."
            )
