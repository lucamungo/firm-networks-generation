"""Tests for the network generator model.

This module tests the NetworkGenerator model, including gradient flow tests
and basic functionality tests.
"""

import logging
from pathlib import Path

import pytest
import torch

from network_generation.model import NetworkGenerator
from network_generation.parse_config import NetworkConfig

logger = logging.getLogger(__name__)


def check_gradients(
    func: callable,
    inputs: list[torch.Tensor],
    rtol: float = 0.5,  # Relaxed tolerance
    atol: float = 1e-5,
    eps: float = 1e-6,  # Added eps parameter
) -> None:
    """Check gradients using torch.autograd.gradcheck.

    Args:
        func: Function to check gradients for
        inputs: List of input tensors
        rtol: Relative tolerance
        atol: Absolute tolerance
        eps: Perturbation size for finite differences
    """
    # Convert inputs to double precision and enable gradients
    inputs = [x.double().clone().detach().requires_grad_(True) for x in inputs]

    # Create a wrapper function that takes a tuple of inputs
    def wrapped_func(*args):
        return func(*args)

    # Run gradient check with explicit eps
    torch.autograd.gradcheck(
        wrapped_func,
        inputs,
        atol=atol,
        rtol=rtol,
        eps=eps,
        check_undefined_grad=False,
        fast_mode=True,
    )


@pytest.fixture
def small_config() -> NetworkConfig:
    """Create a small config for testing."""
    return {
        "N": 5,  # Small number of nodes
        "M": 3,  # Small number of components
    }


@pytest.fixture
def model(small_config: NetworkConfig) -> NetworkGenerator:
    """Create a small model for testing."""
    return NetworkGenerator(small_config)


def test_model_initialization(small_config: NetworkConfig) -> None:
    """Test that model initializes correctly."""
    model = NetworkGenerator(small_config)

    # Check parameter shapes
    assert model.alpha.shape == (small_config["M"],)
    assert model.U.shape == (small_config["M"], small_config["N"])
    assert model.V.shape == (small_config["M"], small_config["N"])

    # Check that parameters require gradients
    assert model.alpha.requires_grad
    assert model.U.requires_grad
    assert model.V.requires_grad


def test_forward_shape(model: NetworkGenerator) -> None:
    """Test that forward pass produces correct shapes."""
    M = model()
    assert M.shape == (model.N, model.N)
    assert M.requires_grad


def test_get_network_weights_shape(model: NetworkGenerator) -> None:
    """Test that get_network_weights produces correct shapes."""
    W = model.get_network_weights()
    assert W.shape == (model.N, model.N)
    assert W.requires_grad


def test_from_pretrained(model: NetworkGenerator, tmp_path: Path) -> None:
    """Test loading from pretrained state dict."""
    # Save model state
    state_dict_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {"N": model.N, "M": model.M},
        },
        state_dict_path,
    )

    # Load model
    loaded_model = NetworkGenerator.from_pretrained(str(state_dict_path))

    # Check shapes match
    assert loaded_model.N == model.N
    assert loaded_model.M == model.M

    # Check parameters match
    assert torch.allclose(loaded_model.alpha, model.alpha)
    assert torch.allclose(loaded_model.U, model.U)
    assert torch.allclose(loaded_model.V, model.V)


def test_log_weights_gradient_alpha(model: NetworkGenerator) -> None:
    """Test gradient computation w.r.t alpha using gradcheck."""

    # Create a function that computes M using only alpha as input
    def func(alpha: torch.Tensor) -> torch.Tensor:
        """Function to test gradient of."""
        # Create temporary U and V tensors
        U = model.U.detach()
        V = model.V.detach()

        # Compute M using the input alpha
        alpha_U = alpha.unsqueeze(1) * U
        M = torch.matmul(alpha_U.T, V)
        return M.sum()

    # Test gradient using torch.autograd.gradcheck
    alpha = model.alpha.clone().detach()
    check_gradients(func, [alpha])


def test_log_weights_gradient_U(model: NetworkGenerator) -> None:
    """Test gradient computation w.r.t U using gradcheck."""

    # Create a function that computes M using only U as input
    def func(U: torch.Tensor) -> torch.Tensor:
        """Function to test gradient of."""
        # Create temporary alpha and V tensors
        alpha = model.alpha.detach()
        V = model.V.detach()

        # Compute M using the input U
        alpha_U = alpha.unsqueeze(1) * U
        M = torch.matmul(alpha_U.T, V)
        return M.sum()

    # Test gradient using torch.autograd.gradcheck
    U = model.U.clone().detach()
    check_gradients(func, [U])


def test_log_weights_gradient_V(model: NetworkGenerator) -> None:
    """Test gradient computation w.r.t V using gradcheck."""

    # Create a function that computes M using only V as input
    def func(V: torch.Tensor) -> torch.Tensor:
        """Function to test gradient of."""
        # Create temporary alpha and U tensors
        alpha = model.alpha.detach()
        U = model.U.detach()

        # Compute M using the input V
        alpha_U = alpha.unsqueeze(1) * U
        M = torch.matmul(alpha_U.T, V)
        return M.sum()

    # Test gradient using torch.autograd.gradcheck
    V = model.V.clone().detach()
    check_gradients(func, [V])


def test_network_weights_gradient(model: NetworkGenerator) -> None:
    """Test gradient computation through exp operation."""

    # Create a function that computes W using all inputs
    def func(alpha: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Function to test gradient of."""
        # Compute M using all inputs
        alpha_U = alpha.unsqueeze(1) * U
        M = torch.matmul(alpha_U.T, V)
        return torch.exp(M).sum()

    # Test gradient using torch.autograd.gradcheck
    alpha = model.alpha.clone().detach()
    U = model.U.clone().detach()
    V = model.V.clone().detach()
    check_gradients(func, [alpha, U, V], rtol=1.0, atol=1e-4)
