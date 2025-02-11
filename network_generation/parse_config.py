"""Configuration parser for network generator.

This module handles parsing and validation of configuration files for the network
generation process. It supports both YAML and JSON formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, TypedDict

import yaml

logger = logging.getLogger(__name__)


class NetworkConfig(TypedDict):
    """Type definition for network configuration."""

    N: int  # Number of nodes
    M: int  # Number of components in matrix factorization
    group_assignments: list[int]  # Group assignment for each node
    correlation_targets: dict[str, float]  # Target correlations
    hill_exponent_targets: dict[str, float]  # Target slopes for log-log CCDF
    io_matrix_target: list[list[float]]  # Target aggregated I/O matrix
    loss_weights: dict[str, float]  # Weights for different loss components
    learning_rate: float
    num_epochs: int
    beta_degree: float  # Softmax temperature for degree computation
    beta_ccdf: float  # Softmax temperature for CCDF computation
    tail_fraction: float  # Fraction of tail to use for Hill exponent
    num_ccdf_points: int  # Number of points to use in CCDF computation


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = {
        "N": int,
        "M": int,
        "group_assignments": list,
        "correlation_targets": dict,
        "hill_exponent_targets": dict,
        "io_matrix_target": list,
        "loss_weights": dict,
        "learning_rate": float,
        "num_epochs": int,
        "beta_degree": float,
        "beta_ccdf": float,
        "tail_fraction": float,
        "num_ccdf_points": int,
    }

    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(config[field], expected_type):
            raise ValueError(
                f"Field {field} has wrong type. Expected {expected_type}, got {type(config[field])}"
            )

    # Additional validation
    if len(config["group_assignments"]) != config["N"]:
        raise ValueError(
            f"group_assignments length ({len(config['group_assignments'])}) "
            f"must match N ({config['N']})"
        )

    required_correlations = {
        "log_in_strength_out_strength",
        "log_in_degree_out_degree",
        "log_out_strength_out_degree",
    }
    if not required_correlations.issubset(config["correlation_targets"].keys()):
        raise ValueError(f"Missing correlation targets. Required: {required_correlations}")

    required_hill_exponents = {"in_degree", "out_degree", "in_strength", "out_strength"}
    if not required_hill_exponents.issubset(config["hill_exponent_targets"].keys()):
        raise ValueError(f"Missing Hill exponent targets. Required: {required_hill_exponents}")


def parse_config(config_path: str | Path) -> NetworkConfig:
    """
    Parse configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing the parsed configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required fields are missing
        ValueError: If field values are invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Required fields and their types
    required_fields = {
        "N": int,  # Number of nodes
        "M": int,  # Number of components in matrix factorization
        "group_assignments": dict,  # Mapping of groups to node indices
        "correlation_targets": {
            "in_out_strength": float,
            "in_out_degree": float,
            "out_strength_degree": float,
        },
        "hill_exponent_targets": {
            "in_degree": float,
            "out_degree": float,
            "in_strength": float,
            "out_strength": float,
        },
        "io_matrix_target": list,  # Will be converted to nested list
        "loss_weights": {"correlation": float, "hill": float, "io": float, "smooth": float},
        "hyperparameters": {
            "learning_rate": float,
            "num_epochs": int,
            "beta_degree": float,
            "beta_ccdf": float,
            "tail_fraction": float,
        },
    }

    # Validate required fields
    _validate_config(config, required_fields)

    return config


def _validate_config(
    config: Dict[str, Any], required_fields: Dict[str, Any], prefix: str = ""
) -> None:
    """
    Recursively validate configuration against required fields specification.

    Args:
        config: Configuration dictionary to validate
        required_fields: Specification of required fields and their types
        prefix: Current path in nested structure for error messages

    Raises:
        KeyError: If a required field is missing
        ValueError: If a field has an invalid type
    """
    for field, field_type in required_fields.items():
        if field not in config:
            full_path = f"{prefix}.{field}" if prefix else field
            raise KeyError(f"Missing required field: {full_path}")

        if isinstance(field_type, dict):
            if not isinstance(config[field], dict):
                raise ValueError(f"Field {field} must be a dictionary")
            _validate_config(config[field], field_type, f"{prefix}.{field}" if prefix else field)
        else:
            if not isinstance(config[field], field_type):
                raise ValueError(
                    f"Field {field} must be of type {field_type.__name__}, "
                    f"got {type(config[field]).__name__}"
                )
