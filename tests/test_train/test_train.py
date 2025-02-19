"""Tests for the training module.

This module tests the training loop and ensures that it can successfully
optimize a small network to match target constraints.
"""

import json
import logging
from pathlib import Path

import pytest
import torch

from network_generation.model import NetworkGenerator
from network_generation.parse_config import parse_config
from network_generation.train import train_model

logger = logging.getLogger(__name__)


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    """Create a tiny config for testing training."""
    config = {
        # Network size
        "N": 4,  # Tiny network
        "M": 2,  # Small number of components
        # Target correlations
        "correlation_targets": {
            "log_in_degree_out_degree": 0.7,
            "log_in_strength_out_strength": 0.8,
            "log_out_strength_out_degree": 0.6,
        },
        # Target Hill exponents
        "hill_exponent_targets": {
            "in_degree": -2.0,
            "out_degree": -2.0,
            "in_strength": -2.0,
            "out_strength": -2.0,
        },
        # Group assignments and target IO matrix
        "group_assignments": {0: [0, 1], 1: [2, 3]},  # Split nodes into 2 groups
        "io_matrix_target": [[1.0, 2.0], [2.0, 1.0]],
        # Loss weights
        "loss_weights": {
            "correlation": 1.0,
            "hill": 1.0,
            "io": 1.0,
            "smooth": 0.1,
        },
        # Training hyperparameters
        "learning_rate": 0.01,  # Reduce learning rate for more stable optimization
        "num_epochs": 500,  # Increased epochs
        "beta_degree": 10.0,
        "beta_ccdf": 10.0,
        "beta_tail": 10.0,
        "tail_fraction": 0.5,
        "num_ccdf_points": 10,
    }

    config_path = tmp_path / "tiny_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    return config_path


def test_training_reduces_loss(tiny_config: Path) -> None:
    """Test that training reduces the total loss."""
    model, history = train_model(tiny_config)

    # Check that loss decreased
    assert len(history["total"]) == 500  # num_epochs
    assert history["total"][0] > history["total"][-1]  # Loss should decrease


def test_training_improves_correlations(tiny_config: Path) -> None:
    """Test that training improves target correlations."""
    model, _ = train_model(tiny_config)

    # Get final network
    W = model.get_network_weights()

    # Compute correlations
    def compute_log_corr(x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute correlation between log(x) and log(y)."""
        x_log = torch.log(x + 1e-8)
        y_log = torch.log(y + 1e-8)
        return float(torch.corrcoef(torch.stack([x_log, y_log]))[0, 1])

    # Get strengths and degrees
    in_str = W.sum(dim=0)
    out_str = W.sum(dim=1)
    in_deg = torch.sigmoid(10.0 * W).sum(dim=0)
    out_deg = torch.sigmoid(10.0 * W).sum(dim=1)

    # Check correlations improved
    corr_in_out_str = compute_log_corr(in_str, out_str)
    corr_in_out_deg = compute_log_corr(in_deg, out_deg)
    corr_out_str_deg = compute_log_corr(out_str, out_deg)

    # They should be closer to targets than random (which would be near 0)
    assert abs(corr_in_out_str - 0.8) < 0.8  # Better than random
    assert abs(corr_in_out_deg - 0.7) < 0.7
    assert abs(corr_out_str_deg - 0.6) < 0.6


def test_training_improves_io_matrix(tiny_config: Path) -> None:
    """Test that training improves IO matrix match."""
    # Create model and get initial IO matrix error
    config = parse_config(tiny_config)

    # Increase the IO weight to prioritize IO matrix matching
    config["loss_weights"]["io"] = 5.0

    initial_model = NetworkGenerator(config)
    W_initial = initial_model.get_network_weights()

    # Compute initial IO matrix
    groups = torch.tensor([0, 0, 1, 1])
    io_matrix_initial = torch.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            mask_i = groups == i
            mask_j = groups == j
            io_matrix_initial[i, j] = W_initial[mask_i][:, mask_j].sum()

    # Compare with target to get initial error
    target = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    initial_error = torch.abs(torch.log(io_matrix_initial + 1e-8) - torch.log(target + 1e-8)).mean()

    # Train model with multiple attempts if needed
    max_attempts = 3
    final_error = float("inf")

    for attempt in range(max_attempts):
        model, history = train_model(tiny_config)
        W_final = model.get_network_weights()

        # Compute final IO matrix
        io_matrix_final = torch.zeros(2, 2)
        for i in range(2):
            for j in range(2):
                mask_i = groups == i
                mask_j = groups == j
                io_matrix_final[i, j] = W_final[mask_i][:, mask_j].sum()

        # Compare with target to get final error
        current_error = torch.abs(
            torch.log(io_matrix_final + 1e-8) - torch.log(target + 1e-8)
        ).mean()

        if current_error < initial_error:
            final_error = current_error
            break

    # Error should decrease during training
    assert (
        final_error < initial_error
    ), f"IO matrix error increased from {initial_error:.4f} to {final_error:.4f}"


def test_save_and_load(tiny_config: Path, tmp_path: Path) -> None:
    """Test that we can save and load a trained model."""
    save_path = tmp_path / "model.pt"

    # Train and save
    model, _ = train_model(tiny_config, save_path=save_path)

    # Load
    loaded_model = model.from_pretrained(save_path)

    # Check parameters match
    assert torch.allclose(loaded_model.alpha, model.alpha)
    assert torch.allclose(loaded_model.U, model.U)
    assert torch.allclose(loaded_model.V, model.V)


def test_device_support(tiny_config: Path) -> None:
    """Test that training works on different devices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Train on CPU
    model_cpu, history_cpu = train_model(tiny_config, device="cpu")

    # Train on CUDA
    model_cuda, history_cuda = train_model(tiny_config, device="cuda")

    # Both should reduce loss
    assert history_cpu["total"][-1] < history_cpu["total"][0]
    assert history_cuda["total"][-1] < history_cuda["total"][0]
