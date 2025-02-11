"""Tests for power-law tail fitting utilities."""

import numpy as np
import pytest
import torch

from network_generation.hill_exponent import (
    compute_ccdf_and_fit_tail,
    compute_hill_exponent,
    hill_loss_from_fit,
)


def generate_power_law(alpha: float, size: int, x_min: float = 1.0, seed: int = 42) -> torch.Tensor:
    """Generate samples from a power-law distribution using inverse transform sampling.

    For a power law P(x) ∝ x^(-alpha), the CCDF is P(X>x) = (x/x_min)^(-(alpha-1))
    Using inverse transform sampling: x = x_min * (1-u)^(-1/(alpha-1)) where u ~ U(0,1)
    """
    torch.manual_seed(seed)
    u = torch.rand(size)
    # Add eps to avoid division by zero when u=1
    x = x_min * ((1 - u + 1e-8) ** (-1.0 / (alpha - 1)))
    return x


def check_gradients(
    func: callable,
    inputs: list[torch.Tensor],
    eps: float = 1e-3,
    rtol: float = 0.5,
    atol: float = 1e-5,
) -> None:
    """Check gradients using torch.autograd.gradcheck.

    Args:
        func: Function to check gradients for
        inputs: List of input tensors
        eps: Perturbation size for finite differences
        rtol: Relative tolerance
        atol: Absolute tolerance
    """
    # Convert inputs to double precision and enable gradients
    inputs = [x.double().clone().detach().requires_grad_(True) for x in inputs]

    # Create a wrapper function that takes a tuple of inputs
    def wrapped_func(*args):
        return func(*args)

    # Run gradient check
    torch.autograd.gradcheck(
        wrapped_func,
        inputs,
        eps=eps,
        atol=atol,
        rtol=rtol,
        check_undefined_grad=False,
        fast_mode=True,
    )


class TestComputeCCDFAndFitTail:
    """Test suite for CCDF computation and tail fitting."""

    def test_power_law_recovery(self) -> None:
        """Test recovery of known power-law exponent."""
        # Generate power law with alpha=2.5 (slope should be around -1.5)
        x = generate_power_law(alpha=2.5, size=5000)

        slope, intercept, ss_res = compute_ccdf_and_fit_tail(
            x, tail_fraction=0.3, num_points=20, beta_ccdf=10.0
        )

        # Check slope is close to expected value
        assert torch.abs(slope + 1.5) < 0.4, f"Expected slope ~ -1.5, got {slope.item()}"

        # Check residuals are small (good line fit)
        assert ss_res < 0.2, f"Expected small residuals, got {ss_res.item()}"

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        x = torch.exp(torch.randn(100)).clamp(0.5, 3.0)

        def func(x_: torch.Tensor) -> torch.Tensor:
            slope, intercept, ss_res = compute_ccdf_and_fit_tail(
                x_, tail_fraction=0.3, num_points=10, beta_ccdf=1.0
            )
            return slope  # Test gradient of slope

        check_gradients(func, [x])

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        x = torch.randn(10)

        # Invalid tail fraction
        with pytest.raises(ValueError):
            compute_ccdf_and_fit_tail(x, tail_fraction=0.0, num_points=5, beta_ccdf=1.0)

        with pytest.raises(ValueError):
            compute_ccdf_and_fit_tail(x, tail_fraction=1.5, num_points=5, beta_ccdf=1.0)

        # Too few points
        with pytest.raises(ValueError):
            compute_ccdf_and_fit_tail(x, tail_fraction=0.2, num_points=1, beta_ccdf=1.0)

        # Non-1D tensor
        with pytest.raises(ValueError):
            compute_ccdf_and_fit_tail(
                x.reshape(2, 5), tail_fraction=0.2, num_points=5, beta_ccdf=1.0
            )

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_ccdf_and_fit_tail(x.numpy(), tail_fraction=0.2, num_points=5, beta_ccdf=1.0)


class TestHillLossFromFit:
    """Test suite for Hill loss computation."""

    def test_perfect_match(self) -> None:
        """Test loss computation with perfect slope match."""
        slope = torch.tensor(-1.5)
        intercept = torch.tensor(0.0)
        ss_res = torch.tensor(0.0)
        target = -1.5
        lambda_line = 1.0

        loss = hill_loss_from_fit(slope, intercept, ss_res, target, lambda_line)
        assert loss < 0.1, f"Expected small loss, got {loss.item()}"

    def test_slope_mismatch(self) -> None:
        """Test loss computation with slope mismatch."""
        slope = torch.tensor(-1.5)
        intercept = torch.tensor(0.0)
        ss_res = torch.tensor(0.0)
        target = -2.0
        lambda_line = 1.0

        loss = hill_loss_from_fit(slope, intercept, ss_res, target, lambda_line)
        expected = (slope - target) ** 2 + 0.01 * intercept**2
        assert torch.allclose(loss, expected)

    def test_line_deviation(self) -> None:
        """Test loss computation with line deviation."""
        slope = torch.tensor(-1.5)
        intercept = torch.tensor(0.0)
        ss_res = torch.tensor(2.0)
        target = -1.5
        lambda_line = 0.5

        loss = hill_loss_from_fit(slope, intercept, ss_res, target, lambda_line)
        expected = lambda_line * ss_res + 0.01 * intercept**2
        assert torch.allclose(loss, expected)

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        slope = torch.tensor(-1.5, requires_grad=True)
        intercept = torch.tensor(1.0, requires_grad=True)
        ss_res = torch.tensor(1.0, requires_grad=True)
        target = -2.0
        lambda_line = 0.5

        def func(s: torch.Tensor, i: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
            return hill_loss_from_fit(s, i, r, target, lambda_line)

        check_gradients(func, [slope, intercept, ss_res])


class TestComputeHillExponent:
    """Test suite for Hill exponent computation."""

    def test_power_law_recovery(self) -> None:
        """Test recovery of known power-law exponent."""
        # Generate power law with known exponent
        true_alpha = 2.5
        x = generate_power_law(alpha=true_alpha, size=5000)

        # Compute Hill exponent - should give (α-1)
        hill_estimate = compute_hill_exponent(x, tail_fraction=0.3, beta_tail=10.0)
        expected_hill = true_alpha - 1  # Hill estimator gives α-1

        # Check if recovered estimate is close to expected value
        assert (
            torch.abs(hill_estimate - expected_hill) < 0.4
        ), f"Expected Hill estimate ≈ {expected_hill}, got {hill_estimate.item()}"

    def test_temperature_effect(self) -> None:
        """Test effect of temperature parameter on estimation."""
        torch.manual_seed(42)
        x = generate_power_law(alpha=2.5, size=5000)
        expected_hill = 1.5  # α-1 = 2.5-1 = 1.5

        # Compare different temperatures
        alpha_low_temp = compute_hill_exponent(x, tail_fraction=0.3, beta_tail=2.0)
        alpha_high_temp = compute_hill_exponent(x, tail_fraction=0.3, beta_tail=0.5)

        # Higher temperature (lower beta) should give less accurate estimates
        assert torch.abs(alpha_high_temp - expected_hill) >= torch.abs(
            alpha_low_temp - expected_hill
        ), f"Lower temperature should give more accurate estimates, getting {alpha_low_temp.item():.3f} and {alpha_high_temp.item():.3f}"

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        x = torch.exp(torch.randn(100)).clamp(0.5, 3.0)

        def func(x_: torch.Tensor) -> torch.Tensor:
            return compute_hill_exponent(x_, tail_fraction=0.3, beta_tail=1.0)

        check_gradients(func, [x])

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        x = torch.randn(10)

        # Invalid tail fraction
        with pytest.raises(ValueError):
            compute_hill_exponent(x, tail_fraction=0.0)

        with pytest.raises(ValueError):
            compute_hill_exponent(x, tail_fraction=1.5)

        # Non-1D tensor
        with pytest.raises(ValueError):
            compute_hill_exponent(x.reshape(2, 5))

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_hill_exponent(x.numpy())
