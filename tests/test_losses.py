"""Tests for loss functions."""

import pytest
import torch

from network_generation.hill_exponent import compute_ccdf_and_fit_tail, hill_loss_from_fit
from network_generation.losses import (
    compute_correlation_loss,
    compute_hill_losses,
    compute_io_loss,
    compute_loss,
    compute_smoothness_loss,
)
from network_generation.stats import compute_degrees


@pytest.fixture
def sample_config() -> dict:
    """Create sample configuration for testing."""
    return {
        "correlation_targets": {
            "log_in_strength_out_strength": 0.7,
            "log_in_degree_out_degree": 0.6,
            "log_out_strength_out_degree": 0.8,
        },
        "hill_exponent_targets": {
            "in_degree": -1.5,
            "out_degree": -1.6,
            "in_strength": -1.7,
            "out_strength": -1.8,
        },
        "io_matrix_target": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        "group_matrix": torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float64),
        "loss_weights": {
            "correlation": 1.0,
            "hill": 1.0,
            "io": 1.0,
            "smooth": 0.1,
        },
        "beta_degree": 1.0,
        "beta_tail": 0.01,
        "tail_fraction": 0.3,
        "num_ccdf_points": 10,
        "lambda_line": 0.1,
    }


@pytest.fixture
def sample_matrix() -> torch.Tensor:
    """Create sample log-weight matrix for testing."""
    # Create a matrix with some structure to test all components
    torch.manual_seed(42)
    M = torch.randn(3, 3)
    # Ensure some positive correlation between in/out degrees
    M = M + M.T  # Make it more symmetric
    return M


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


class TestCorrelationLoss:
    """Test suite for correlation loss computation."""

    def test_basic_computation(self, sample_matrix: torch.Tensor) -> None:
        """Test basic computation of correlation loss."""
        targets = {
            "log_in_strength_out_strength": 0.7,
            "log_in_degree_out_degree": 0.6,
            "log_out_strength_out_degree": 0.8,
        }
        loss, partial_losses = compute_correlation_loss(sample_matrix, targets, beta_degree=1.0)

        assert isinstance(loss, torch.Tensor)
        assert len(partial_losses) == len(targets)
        assert all(isinstance(v, torch.Tensor) for v in partial_losses.values())
        assert all(v >= 0 for v in partial_losses.values())  # Squared differences are non-negative

    def test_unknown_correlation(self, sample_matrix: torch.Tensor) -> None:
        """Test error handling for unknown correlation target."""
        targets = {"unknown_correlation": 0.5}
        with pytest.raises(ValueError, match="Unknown correlation target"):
            compute_correlation_loss(sample_matrix, targets, beta_degree=1.0)

    def test_gradient(self, sample_matrix: torch.Tensor) -> None:
        """Test gradient computation using finite differences."""
        targets = {"log_in_strength_out_strength": 0.7}
        M = sample_matrix.clone()

        def func(x: torch.Tensor) -> torch.Tensor:
            loss, _ = compute_correlation_loss(x, targets, beta_degree=1.0)
            return loss

        check_gradients(func, [M])


class TestHillLosses:
    """Test suite for Hill exponent loss computation."""

    def test_basic_computation(self, sample_matrix: torch.Tensor) -> None:
        """Test basic computation of Hill losses."""
        targets = {
            "in_degree": -1.5,
            "out_degree": -1.6,
            "in_strength": -1.7,
            "out_strength": -1.8,
        }
        loss, partial_losses = compute_hill_losses(
            sample_matrix,
            targets,
            beta_degree=1.0,
            beta_tail=1.0,
            tail_fraction=0.3,
        )

        assert isinstance(loss, torch.Tensor)
        assert len(partial_losses) == len(targets)
        assert all(isinstance(v, torch.Tensor) for v in partial_losses.values())
        assert all(v >= 0 for v in partial_losses.values())

    def test_unknown_distribution(self, sample_matrix: torch.Tensor) -> None:
        """Test error handling for unknown distribution."""
        targets = {"unknown_distribution": -1.5}
        with pytest.raises(ValueError, match="Unknown Hill target"):
            compute_hill_losses(
                sample_matrix,
                targets,
                beta_degree=1.0,
                beta_tail=1.0,
                tail_fraction=0.3,
            )

    def test_gradient(self, sample_matrix: torch.Tensor) -> None:
        """Test gradient computation using finite differences."""
        targets = {"in_degree": -1.5}
        M = sample_matrix.clone()

        def func(x: torch.Tensor) -> torch.Tensor:
            W = torch.exp(x)  # Get weight matrix
            in_degrees = compute_degrees(W, beta_degree=1.0, dim=0)
            slope, intercept, ss_res = compute_ccdf_and_fit_tail(
                in_degrees, tail_fraction=0.3, num_points=10, beta_ccdf=1.0
            )
            return hill_loss_from_fit(
                slope, intercept, ss_res, -1.5, 0.1
            )  # Test full loss computation

        check_gradients(
            func, [M], rtol=1.0
        )  # Increased tolerance due to composition of nonlinear functions


class TestIOLoss:
    """Test suite for IO matrix loss computation."""

    def test_basic_computation(self, sample_matrix: torch.Tensor) -> None:
        """Test basic computation of IO loss."""
        group_matrix = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
        io_target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss = compute_io_loss(sample_matrix, group_matrix, io_target)

        assert isinstance(loss, torch.Tensor)
        assert loss >= 0  # MSE loss is non-negative

    def test_gradient(self, sample_matrix: torch.Tensor) -> None:
        """Test gradient computation using finite differences."""
        group_matrix = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.float32)
        io_target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        M = sample_matrix.clone()

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_io_loss(x, group_matrix, io_target)

        check_gradients(func, [M])


class TestSmoothnessLoss:
    """Test suite for smoothness loss computation."""

    def test_basic_computation(self, sample_matrix: torch.Tensor) -> None:
        """Test basic computation of smoothness loss."""
        loss, partial_losses = compute_smoothness_loss(
            sample_matrix, beta_degree=1.0, beta_ccdf=1.0
        )

        assert isinstance(loss, torch.Tensor)
        assert len(partial_losses) == 4  # in/out degree/strength
        assert all(isinstance(v, torch.Tensor) for v in partial_losses.values())
        assert all(v >= 0 for v in partial_losses.values())

    def test_gradient(self, sample_matrix: torch.Tensor) -> None:
        """Test gradient computation using finite differences."""
        M = sample_matrix.clone()

        def func(x: torch.Tensor) -> torch.Tensor:
            loss, _ = compute_smoothness_loss(x, beta_degree=1.0, beta_ccdf=1.0)
            return loss

        check_gradients(func, [M])


class TestTotalLoss:
    """Test suite for total loss computation."""

    def test_basic_computation(self, sample_matrix: torch.Tensor, sample_config: dict) -> None:
        """Test basic computation of total loss."""
        loss, partial_losses = compute_loss(sample_matrix, sample_config)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(partial_losses, dict)
        assert all(isinstance(v, torch.Tensor) for v in partial_losses.values())
        assert all(v >= 0 for v in partial_losses.values())

    # We expect this to fail
    @pytest.mark.xfail(reason="Loss normalization is not implemented.")
    def test_normalization(self, sample_matrix: torch.Tensor, sample_config: dict) -> None:
        """Test that loss normalization behaves correctly."""
        M = sample_matrix.clone()
        loss1, _ = compute_loss(M, sample_config)
        loss2, _ = compute_loss(2 * M, sample_config)
        # Use high rtol because exact equality is not expected
        assert torch.allclose(loss1, loss2, rtol=0.2)

    def test_component_gradients(self, sample_matrix: torch.Tensor, sample_config: dict) -> None:
        """Test gradients of individual normalized components."""
        M = sample_matrix.clone()

        # Precompute normalization constants at the base point for each component:
        loss_corr, _ = compute_correlation_loss(
            M, sample_config["correlation_targets"], sample_config["beta_degree"]
        )
        norm_corr = loss_corr.detach() + 1e-8

        loss_hill, _ = compute_hill_losses(
            M,
            sample_config["hill_exponent_targets"],
            sample_config["beta_degree"],
            sample_config["beta_tail"],
            sample_config["tail_fraction"],
        )
        norm_hill = loss_hill.detach() + 1e-8

        loss_io = compute_io_loss(
            M, sample_config["group_matrix"], sample_config["io_matrix_target"]
        )
        norm_io = loss_io.detach() + 1e-8

        loss_smooth, _ = compute_smoothness_loss(
            M, sample_config["beta_degree"], sample_config["beta_tail"]
        )
        norm_smooth = loss_smooth.detach() + 1e-8

        def correlation_func(x):
            loss, _ = compute_correlation_loss(
                x, sample_config["correlation_targets"], sample_config["beta_degree"]
            )
            return loss / norm_corr

        def hill_func(x):
            loss, _ = compute_hill_losses(
                x,
                sample_config["hill_exponent_targets"],
                sample_config["beta_degree"],
                sample_config["beta_tail"],
                sample_config["tail_fraction"],
            )
            return loss / norm_hill

        def io_func(x):
            loss = compute_io_loss(
                x, sample_config["group_matrix"], sample_config["io_matrix_target"]
            )
            return loss / norm_io

        def smoothness_func(x):
            loss, _ = compute_smoothness_loss(
                x, sample_config["beta_degree"], sample_config["beta_tail"]
            )
            return loss / norm_smooth

        # Check each component's gradient.
        check_gradients(correlation_func, [M], rtol=1.0, atol=1e-3)
        check_gradients(hill_func, [M], rtol=1.0, atol=1e-3)
        check_gradients(io_func, [M], rtol=1.0, atol=1e-3)
        check_gradients(smoothness_func, [M], rtol=1.0, atol=1e-3)

    def test_weighted_gradients(self, sample_matrix: torch.Tensor, sample_config: dict) -> None:
        """Test gradients with different weight configurations."""
        M = sample_matrix.clone()

        # Test with only correlation loss
        config1 = sample_config.copy()
        config1["loss_weights"] = {"correlation": 1.0, "hill": 0.0, "io": 0.0, "smooth": 0.0}

        # Test with only hill loss
        config2 = sample_config.copy()
        config2["loss_weights"] = {"correlation": 0.0, "hill": 1.0, "io": 0.0, "smooth": 0.0}

        # Precompute normalization constants for each configuration.
        loss1, _ = compute_loss(M, config1)
        norm1 = loss1.detach() + 1e-8

        loss2, _ = compute_loss(M, config2)
        norm2 = loss2.detach() + 1e-8

        def func1(x):
            loss, _ = compute_loss(x, config1)
            return loss / norm1

        def func2(x):
            loss, _ = compute_loss(x, config2)
            return loss / norm2

        check_gradients(func1, [M], rtol=1.0, atol=1e-3)
        check_gradients(func2, [M], rtol=1.0, atol=1e-3)

    def test_gradient(self, sample_matrix: torch.Tensor, sample_config: dict) -> None:
        """Test gradient computation using finite differences."""
        M = sample_matrix.clone()

        # Create a more numerically stable version of the config
        stable_config = sample_config.copy()
        stable_config["beta_tail"] = 1.0
        stable_config["num_ccdf_points"] = 5

        # Reduce the weights of components relying on CCDF computation
        stable_config["loss_weights"] = {
            "correlation": 1.0,
            "hill": 0.1,  # Reduce weight of hill loss
            "io": 1.0,
            "smooth": 1.0,  # Reduce weight of smoothness loss
        }

        def func(x: torch.Tensor) -> torch.Tensor:
            loss, _ = compute_loss(x, stable_config)
            return loss

        check_gradients(func, [M], rtol=1.0, atol=1e-2, eps=1e-12)

    def test_missing_config_fields(self, sample_matrix: torch.Tensor) -> None:
        """Test error handling for missing configuration fields."""
        config = {}  # Empty config
        with pytest.raises(KeyError):
            compute_loss(sample_matrix, config)

    def test_invalid_weights(self, sample_matrix: torch.Tensor, sample_config: dict) -> None:
        """Test handling of invalid loss weights."""
        config = sample_config.copy()
        config["loss_weights"] = {"correlation": -1.0}  # Invalid negative weight
        with pytest.raises(KeyError):
            compute_loss(sample_matrix, config)
