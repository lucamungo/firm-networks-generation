"""Tests for configuration parsing utilities."""

from pathlib import Path

import pytest
import yaml

from network_generation.parse_config import parse_config


@pytest.fixture
def valid_config() -> dict:
    """Create a valid configuration dictionary for testing."""
    # Create group assignments for all 100 nodes
    group_assignments = {
        "industry_1": list(range(0, 50)),  # First 50 nodes
        "industry_2": list(range(50, 100)),  # Last 50 nodes
    }

    return {
        "N": 100,
        "M": 5,
        "group_assignments": group_assignments,
        "correlation_targets": {
            "log_in_strength_out_strength": 0.7,
            "log_in_degree_out_degree": 0.6,
            "log_out_strength_out_degree": 0.8,
        },
        "hill_exponent_targets": {
            "in_degree": 2.1,
            "out_degree": 2.2,
            "in_strength": 2.3,
            "out_strength": 2.4,
        },
        "io_matrix_target": [[1.0, 2.0], [3.0, 4.0]],
        "loss_weights": {"correlation": 1.0, "hill": 1.0, "io": 1.0, "smooth": 0.1},
        "learning_rate": 0.001,
        "num_epochs": 1000,
        "beta_degree": 10.0,
        "beta_ccdf": 10.0,
        "tail_fraction": 0.2,
        "num_ccdf_points": 20,
    }


@pytest.fixture
def config_file(tmp_path: Path, valid_config: dict) -> Path:
    """Create a temporary config file with valid configuration."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(valid_config, f)
    return config_path


def test_parse_valid_config(config_file: Path) -> None:
    """Test parsing of a valid configuration file."""
    config = parse_config(config_file)
    assert config["N"] == 100
    assert config["M"] == 5
    assert len(config["group_assignments"]) == 2
    assert config["correlation_targets"]["log_in_strength_out_strength"] == 0.7
    assert config["hill_exponent_targets"]["in_degree"] == 2.1
    assert len(config["io_matrix_target"]) == 2
    assert config["loss_weights"]["correlation"] == 1.0
    assert config["learning_rate"] == 0.001


def test_missing_file() -> None:
    """Test error handling for non-existent config file."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        parse_config("nonexistent.yaml")


def test_missing_required_field(tmp_path: Path, valid_config: dict) -> None:
    """Test error handling for missing required field."""
    del valid_config["N"]
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(valid_config, f)

    with pytest.raises(KeyError, match="Missing required field: N"):
        parse_config(config_path)


def test_invalid_field_type(tmp_path: Path, valid_config: dict) -> None:
    """Test error handling for invalid field type."""
    valid_config["N"] = "not_an_integer"
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(valid_config, f)

    with pytest.raises(ValueError, match="Field N must be of type int"):
        parse_config(config_path)


def test_missing_nested_field(tmp_path: Path, valid_config: dict) -> None:
    """Test error handling for missing nested field."""
    del valid_config["correlation_targets"]["log_in_strength_out_strength"]
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(valid_config, f)

    with pytest.raises(ValueError):
        parse_config(config_path)


def test_invalid_nested_field_type(tmp_path: Path, valid_config: dict) -> None:
    """Test error handling for invalid nested field type."""
    valid_config["correlation_targets"]["log_in_strength_out_strength"] = "not_a_float"
    config_path = tmp_path / "invalid_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(valid_config, f)

    with pytest.raises(ValueError):
        parse_config(config_path)
