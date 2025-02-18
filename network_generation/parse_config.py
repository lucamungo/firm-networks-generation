"""Configuration parser for network generator.

This module handles parsing and validation of configuration files for the network
generation process. It supports both YAML and JSON formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, TypedDict

import torch
import yaml

logger = logging.getLogger(__name__)


class NetworkConfig(TypedDict):
    """Type definition for network configuration."""

    N: int  # Number of nodes
    M: int  # Number of components in matrix factorization
    group_assignments: dict[str, list[int]]  # Group assignment for each node
    group_matrix: torch.Tensor  # Binary matrix indicating group membership
    correlation_targets: dict[str, float]  # Target correlations
    hill_exponent_targets: dict[str, float]  # Target slopes for log-log CCDF
    io_matrix_target: torch.Tensor  # Target aggregated I/O matrix
    loss_weights: dict[str, float]  # Weights for different loss components
    learning_rate: float
    num_epochs: int
    beta_degree: float  # Softmax temperature for degree computation
    beta_ccdf: float  # Softmax temperature for CCDF computation
    tail_fraction: float  # Fraction of tail to use for Hill exponent
    num_ccdf_points: int  # Number of points to use in CCDF computation


def create_group_matrix(group_assignments: Dict[str, list[int]], N: int) -> torch.Tensor:
    """Create a binary group membership matrix from group assignments.

    Args:
        group_assignments: Dictionary mapping group IDs to lists of node indices
        N: Total number of nodes

    Returns:
        Binary matrix of shape (num_groups, N) where entry (i,j) is 1 if node j
        belongs to group i
    """
    # Get unique group IDs (no need to convert to integers)
    group_ids = sorted(group_assignments.keys())
    num_groups = len(group_ids)

    # Create binary matrix
    group_matrix = torch.zeros((num_groups, N), dtype=torch.float64)
    for i, group_id in enumerate(group_ids):
        for node_idx in group_assignments[group_id]:
            group_matrix[i, node_idx] = 1.0

    return group_matrix


def validate_config(config: dict[str, Any]) -> None:
    """Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValueError: If configuration is invalid
    """
    # First validate top-level fields
    required_fields = {
        "N": int,
        "M": int,
        "group_assignments": dict,
        "correlation_targets": dict,
        "hill_exponent_targets": dict,
        "io_matrix_target": (list, torch.Tensor),  # Allow both list and tensor
        "loss_weights": dict,
    }

    # Add hyperparameter fields
    hyperparameter_fields = {
        "learning_rate": float,
        "num_epochs": int,
        "beta_degree": float,
        "beta_ccdf": float,
        "tail_fraction": float,
        "num_ccdf_points": int,
    }
    required_fields.update(hyperparameter_fields)

    # Validate field presence and types
    for field, expected_type in required_fields.items():
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

        if isinstance(expected_type, tuple):
            if not any(isinstance(config[field], t) for t in expected_type):
                type_names = " or ".join(t.__name__ for t in expected_type)
                raise ValueError(
                    f"Field {field} has wrong type. Expected {type_names}, "
                    f"got {type(config[field]).__name__}"
                )
        elif not isinstance(config[field], expected_type):
            raise ValueError(
                f"Field {field} has wrong type. Expected {expected_type.__name__}, "
                f"got {type(config[field]).__name__}"
            )

    # Validate correlation targets
    required_correlations = {
        "log_in_strength_out_strength",
        "log_in_degree_out_degree",
        "log_out_strength_out_degree",
    }
    missing_correlations = required_correlations - set(config["correlation_targets"].keys())
    if missing_correlations:
        raise ValueError(f"Missing correlation targets: {missing_correlations}")

    # Validate correlation target types
    for name, value in config["correlation_targets"].items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Correlation target {name} has wrong type. Expected number, "
                f"got {type(value).__name__}"
            )

    # Validate Hill exponent targets
    required_hill_exponents = {"in_degree", "out_degree", "in_strength", "out_strength"}
    missing_exponents = required_hill_exponents - set(config["hill_exponent_targets"].keys())
    if missing_exponents:
        raise ValueError(f"Missing Hill exponent targets: {missing_exponents}")

    # Validate Hill exponent target types
    for name, value in config["hill_exponent_targets"].items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Hill exponent target {name} has wrong type. Expected number, "
                f"got {type(value).__name__}"
            )

    # Validate loss weights
    required_weights = {"correlation", "hill", "io", "smooth"}
    missing_weights = required_weights - set(config["loss_weights"].keys())
    if missing_weights:
        raise ValueError(f"Missing loss weights: {missing_weights}")

    # Validate loss weight types
    for name, value in config["loss_weights"].items():
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Loss weight {name} has wrong type. Expected number, "
                f"got {type(value).__name__}"
            )

    # Validate group assignments
    assigned_nodes = set()
    for group_id, nodes in config["group_assignments"].items():
        if not isinstance(nodes, list):
            raise ValueError(
                f"Group {group_id} nodes has wrong type. Expected list, "
                f"got {type(nodes).__name__}"
            )

        # Check that nodes are valid indices
        for node in nodes:
            if not isinstance(node, int):
                raise ValueError(
                    f"Node indices must be integers, got {type(node)} in group {group_id}"
                )
            if node < 0 or node >= config["N"]:
                raise ValueError(
                    f"Node index {node} in group {group_id} out of range [0, {config['N']})"
                )
            if node in assigned_nodes:
                raise ValueError(f"Node {node} assigned to multiple groups")
            assigned_nodes.add(node)

    # Check that all nodes are assigned
    if len(assigned_nodes) != config["N"]:
        raise ValueError(
            f"Not all nodes assigned to groups. Expected {config['N']} nodes, "
            f"but only {len(assigned_nodes)} were assigned."
        )


def parse_config(config_path: str | Path) -> NetworkConfig:
    """
    Parse configuration from a YAML or JSON file.

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

    # Load config based on file extension
    with open(config_path) as f:
        if config_path.suffix == ".yaml":
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    # Extract hyperparameters if they exist in nested dict
    if "hyperparameters" in config:
        for k, v in config["hyperparameters"].items():
            config[k] = v
        del config["hyperparameters"]

    # First validate required fields for group matrix creation
    if "N" not in config:
        raise KeyError("Missing required field: N")
    if not isinstance(config["N"], int):
        raise ValueError(f"Field N must be of type int, got {type(config['N']).__name__}")

    if "group_assignments" not in config:
        raise KeyError("Missing required field: group_assignments")
    if not isinstance(config["group_assignments"], dict):
        raise ValueError(
            f"Field group_assignments must be of type dict, "
            f"got {type(config['group_assignments']).__name__}"
        )

    # Create group matrix from assignments
    config["group_matrix"] = create_group_matrix(config["group_assignments"], config["N"])

    # Convert IO matrix target to tensor if it's a list
    if "io_matrix_target" in config and isinstance(config["io_matrix_target"], list):
        config["io_matrix_target"] = torch.tensor(config["io_matrix_target"], dtype=torch.float64)

    # Set default values for optional fields
    config.setdefault("num_ccdf_points", 20)
    config.setdefault(
        "beta_tail", config.get("beta_ccdf", 10.0)
    )  # Default to beta_ccdf if available

    # Validate the rest of the configuration
    validate_config(config)

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
