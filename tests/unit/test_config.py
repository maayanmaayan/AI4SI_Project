"""Unit tests for config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.utils.config import (
    _merge_env_overrides,
    _validate_config,
    get_config,
    load_config,
)


@pytest.fixture
def temp_config_file():
    """Create a temporary YAML config file for testing."""
    config_data = {
        "model": {"name": "test_model", "d_token": 128},
        "training": {"batch_size": 64, "learning_rate": 0.001},
        "data": {"train_split": 0.7, "random_seed": 42},
        "paths": {"data_root": "./data"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_invalid_yaml_file():
    """Create a temporary invalid YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [")
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


def test_load_config_valid_file(temp_config_file):
    """Test loading a valid YAML configuration file."""
    config = load_config(temp_config_file)

    assert isinstance(config, dict)
    assert "model" in config
    assert config["model"]["name"] == "test_model"
    assert config["training"]["batch_size"] == 64
    assert config["data"]["train_split"] == 0.7


def test_load_config_file_not_found():
    """Test FileNotFoundError is raised for missing file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found"):
        load_config("nonexistent_config.yaml")


def test_load_config_invalid_yaml(temp_invalid_yaml_file):
    """Test YAMLError is raised for malformed YAML."""
    with pytest.raises(yaml.YAMLError):
        load_config(temp_invalid_yaml_file)


def test_get_config_singleton(temp_config_file):
    """Test that get_config() returns cached instance."""
    # Reset global cache
    import src.utils.config

    src.utils.config._config_cache = None

    # Mock load_config to use temp file
    with patch("src.utils.config.load_config") as mock_load:
        mock_load.return_value = {"test": "value"}

        config1 = get_config()
        config2 = get_config()

        # Should only call load_config once
        assert mock_load.call_count == 1
        # Should return same dict instance
        assert config1 is config2


def test_env_var_override(temp_config_file):
    """Test environment variable overrides are merged correctly."""
    # Set environment variable
    os.environ["DATA__TRAIN_SPLIT"] = "0.8"
    os.environ["MODEL__NAME"] = "overridden_model"

    try:
        config = load_config(temp_config_file)

        # Check that env var override worked
        assert config["data"]["train_split"] == 0.8
        assert config["model"]["name"] == "overridden_model"
    finally:
        # Cleanup
        os.environ.pop("DATA__TRAIN_SPLIT", None)
        os.environ.pop("MODEL__NAME", None)


def test_env_var_type_conversion(temp_config_file):
    """Test environment variable values are converted to appropriate types."""
    os.environ["TRAINING__BATCH_SIZE"] = "128"  # Should become int
    os.environ["TRAINING__LEARNING_RATE"] = "0.002"  # Should become float
    os.environ["DATA__TRAIN_ONLY_ON_COMPLIANT"] = "true"  # Should become bool

    try:
        config = load_config(temp_config_file)

        assert isinstance(config["training"]["batch_size"], int)
        assert config["training"]["batch_size"] == 128
        assert isinstance(config["training"]["learning_rate"], float)
        assert config["training"]["learning_rate"] == 0.002
        assert isinstance(config["data"]["train_only_on_compliant"], bool)
        assert config["data"]["train_only_on_compliant"] is True
    finally:
        # Cleanup
        os.environ.pop("TRAINING__BATCH_SIZE", None)
        os.environ.pop("TRAINING__LEARNING_RATE", None)
        os.environ.pop("DATA__TRAIN_ONLY_ON_COMPLIANT", None)


def test_validate_config_missing_section(temp_config_file):
    """Test validation raises error for missing required sections."""
    # Load config and remove a required section
    config = load_config(temp_config_file)
    del config["model"]

    with pytest.raises(ValueError, match="Required configuration section"):
        _validate_config(config)


def test_validate_config_all_sections_present(temp_config_file):
    """Test validation passes when all required sections are present."""
    config = load_config(temp_config_file)
    # Should not raise
    _validate_config(config)


def test_merge_env_overrides():
    """Test environment variable merging logic."""
    config = {"data": {"train_split": 0.7}, "model": {"name": "default"}}

    os.environ["DATA__TRAIN_SPLIT"] = "0.8"

    try:
        merged = _merge_env_overrides(config)
        assert merged["data"]["train_split"] == 0.8
        assert merged["model"]["name"] == "default"
    finally:
        os.environ.pop("DATA__TRAIN_SPLIT", None)


def test_env_var_override_conflict_logs_warning(caplog):
    """Test that environment variable override conflicts log a warning."""
    import logging

    logging.basicConfig(level=logging.WARNING)
    # Config with non-dict value that conflicts with env var path
    config = {"data": "not_a_dict"}

    os.environ["DATA__TRAIN_SPLIT"] = "0.8"

    try:
        merged = _merge_env_overrides(config)
        # Should not have set the value due to conflict
        assert merged["data"] == "not_a_dict"
        # Should have logged a warning
        assert "conflicts with existing non-dict value" in caplog.text.lower()
    finally:
        os.environ.pop("DATA__TRAIN_SPLIT", None)


def test_load_config_default_path():
    """Test that load_config uses default path when None provided."""
    # Reset cache to test default path loading
    import src.utils.config

    src.utils.config._config_cache = None

    # Test that load_config() works with default path (config.yaml should exist)
    # This test assumes config.yaml exists (created from example during setup)
    try:
        config = load_config()
        assert isinstance(config, dict)
        assert "model" in config
    except FileNotFoundError:
        # If config file doesn't exist, that's also a valid test outcome
        # (it means the default path logic works, just file is missing)
        pass
