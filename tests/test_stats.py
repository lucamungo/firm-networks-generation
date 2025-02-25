"""Tests for statistical computation functions."""

import numpy as np
import pytest
import torch

from network_generation.stats import (
    compute_ccdf,
    compute_degrees,
    compute_density,
    compute_io_matrix,
    compute_log_correlation,
    compute_smoothness_penalty,
    compute_strengths,
)


@pytest.fixture
def sample_matrix() -> torch.Tensor:
    """Create a sample weighted adjacency matrix for testing."""
    return torch.tensor([[0.0, 2.0, 3.0], [4.0, 0.0, 6.0], [7.0, 8.0, 0.0]], dtype=torch.float32)


@pytest.fixture
def sample_group_matrix() -> torch.Tensor:
    """Create a sample group assignment matrix for testing."""
    return torch.tensor(
        [[1, 1, 0], [0, 0, 1]],  # First group: nodes 0,1  # Second group: node 2
        dtype=torch.float32,
    )


@pytest.fixture
def controlled_matrix() -> torch.Tensor:
    """Create a controlled-range matrix to avoid extreme saturation."""
    torch.manual_seed(42)  # For reproducibility
    # Use smaller range [-2, 2] to avoid sigmoid saturation
    return torch.randn(4, 4).clamp(-2.0, 2.0)


def check_gradients(
    func: callable,
    inputs: list[torch.Tensor],
    eps: float = 1e-4,  # Increased from 1e-6 for better numerical stability
    rtol: float = 5e-2,  # Relaxed relative tolerance
    atol: float = 1e-3,  # Absolute tolerance for near-zero gradients
    skip_small: bool = False,  # Option to skip checking very small gradients
) -> None:
    """
    Check gradients using finite differences with robust handling of nonlinear functions.

    Args:
        func: Function that takes a list of tensors and returns a scalar
        inputs: List of input tensors
        eps: Step size for finite difference
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison
        skip_small: Whether to skip checking very small gradients
    """
    # Enable gradients
    inputs = [x.clone().detach().requires_grad_(True) for x in inputs]

    # Compute analytical gradients
    output = func(*inputs)
    output.backward()
    analytical_grads = [x.grad.clone() for x in inputs]

    # Zero out gradients
    for x in inputs:
        x.grad.zero_()

    # Compute numerical gradients for each input
    for i, x in enumerate(inputs):
        numerical_grad = torch.zeros_like(x)

        # Compute gradient for each element
        for idx in range(x.numel()):
            # Create new tensors for positive and negative perturbations
            inputs_pos = [t.clone() if j != i else t.clone().detach() for j, t in enumerate(inputs)]
            inputs_neg = [t.clone() if j != i else t.clone().detach() for j, t in enumerate(inputs)]

            # Apply perturbations
            inputs_pos[i].view(-1)[idx] += eps
            inputs_neg[i].view(-1)[idx] -= eps

            # Compute function values
            pos_out = func(*inputs_pos).item()
            neg_out = func(*inputs_neg).item()

            # Two-sided finite difference
            numerical_grad.view(-1)[idx] = (pos_out - neg_out) / (2 * eps)

        # Compare gradients
        analytical_grad = analytical_grads[i]

        # Handle both relative and absolute errors
        abs_diff = torch.abs(analytical_grad - numerical_grad)
        abs_numerical = torch.abs(numerical_grad)

        if skip_small:
            # Only check gradients above a certain threshold
            significant = abs_numerical > 1e-2
        else:
            # Check all gradients but with different tolerances
            significant = abs_numerical > atol

        # Use relative error where numerical gradient is significant
        if significant.any():
            rel_error = abs_diff[significant] / abs_numerical[significant]
            assert torch.all(
                rel_error < rtol
            ), f"Relative error too large: {rel_error.max().item()}"

        # Use absolute error where numerical gradient is small
        small = ~significant
        if small.any() and not skip_small:
            assert torch.all(
                abs_diff[small] < atol
            ), f"Absolute error too large: {abs_diff[small].max().item()}"

        # Check correlation for significant gradients
        if significant.any():
            analytical_sig = analytical_grad[significant]
            numerical_sig = numerical_grad[significant]
            norm_analytical = torch.norm(analytical_sig)
            norm_numerical = torch.norm(numerical_sig)
            if norm_analytical > atol and norm_numerical > atol:
                cosine = torch.sum(analytical_sig * numerical_sig) / (
                    norm_analytical * norm_numerical
                )
                assert (
                    cosine > 0.9
                ), f"Gradient directions differ too much: cosine = {cosine.item()}"


class TestComputeDegrees:
    """Test suite for degree computation."""

    def test_in_degrees(self, sample_matrix: torch.Tensor) -> None:
        """Test computation of in-degrees."""
        degrees = compute_degrees(sample_matrix, beta_degree=1.0, dim=0)
        # For each column, we expect sigmoid(values).sum()
        expected = (
            torch.tensor(
                [
                    torch.sigmoid(torch.tensor([0.0, 4.0, 7.0])).sum(),
                    torch.sigmoid(torch.tensor([2.0, 0.0, 8.0])).sum(),
                    torch.sigmoid(torch.tensor([3.0, 6.0, 0.0])).sum(),
                ]
            )
            + 1e-8
        )
        assert torch.allclose(degrees, expected)

    def test_out_degrees(self, sample_matrix: torch.Tensor) -> None:
        """Test computation of out-degrees."""
        degrees = compute_degrees(sample_matrix, beta_degree=1.0, dim=1)
        # For each row, we expect sigmoid(values).sum()
        expected = (
            torch.tensor(
                [
                    torch.sigmoid(torch.tensor([0.0, 2.0, 3.0])).sum(),
                    torch.sigmoid(torch.tensor([4.0, 0.0, 6.0])).sum(),
                    torch.sigmoid(torch.tensor([7.0, 8.0, 0.0])).sum(),
                ]
            )
            + 1e-8
        )
        assert torch.allclose(degrees, expected)

    def test_beta_effect(self, sample_matrix: torch.Tensor) -> None:
        """Test effect of beta parameter on degree computation."""
        degrees_low_beta = compute_degrees(sample_matrix, beta_degree=0.1)
        degrees_high_beta = compute_degrees(sample_matrix, beta_degree=10.0)
        # Higher beta should make sigmoid sharper, leading to more extreme values
        assert torch.all(torch.abs(degrees_high_beta - 1.5) > torch.abs(degrees_low_beta - 1.5))

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError):
            compute_degrees(torch.randn(3, 4), beta_degree=1.0)

        # Invalid dimension
        with pytest.raises(ValueError):
            compute_degrees(torch.randn(3, 3), beta_degree=1.0, dim=2)

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_degrees(np.random.randn(3, 3), beta_degree=1.0)

    def test_gradient(self, controlled_matrix: torch.Tensor) -> None:
        """Test gradient computation using finite differences."""
        W = controlled_matrix  # Use controlled range matrix
        beta = 1.0

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_degrees(x, beta, dim=1).sum()

        check_gradients(func, [W], skip_small=True)  # Skip very small gradients


class TestComputeStrengths:
    """Test suite for strength computation."""

    def test_in_strengths(self, sample_matrix: torch.Tensor) -> None:
        """Test computation of in-strengths."""
        strengths = compute_strengths(sample_matrix, dim=0)
        expected = torch.tensor([11.0, 10.0, 9.0]) + 1e-8  # Column sums
        assert torch.allclose(strengths, expected)

    def test_out_strengths(self, sample_matrix: torch.Tensor) -> None:
        """Test computation of out-strengths."""
        strengths = compute_strengths(sample_matrix, beta=10, dim=1)
        expected = torch.tensor([5.0, 10.0, 15.0]) + 1e-8  # Row sums
        assert torch.allclose(strengths, expected)

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError):
            compute_strengths(torch.randn(3, 4))

        # Invalid dimension
        with pytest.raises(ValueError):
            compute_strengths(torch.randn(3, 3), dim=2)

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_strengths(np.random.randn(3, 3))

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        W = torch.randn(4, 4)

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_strengths(x, dim=1).sum()

        check_gradients(func, [W])


class TestComputeDensity:
    """Test suite for density computation."""

    def test_basic_computation(self, sample_matrix: torch.Tensor) -> None:
        """Test basic computation of density."""
        density = compute_density(sample_matrix)
        assert isinstance(density, torch.Tensor)
        assert 0 <= density <= 1  # Density should be between 0 and 1
        assert not torch.isnan(density)  # Density should not be NaN
        assert not torch.isinf(density)  # Density should not be infinite

    def test_beta_effect(self, sample_matrix: torch.Tensor) -> None:
        """Test effect of beta parameter on density computation."""
        # Low beta should give more "fuzzy" density
        density_low_beta = compute_density(sample_matrix, beta=0.1)

        # High beta should give more "sharp" density
        density_high_beta = compute_density(sample_matrix, beta=10.0)

        assert (
            density_low_beta != density_high_beta
        )  # Different betas should give different densities
        assert 0 <= density_low_beta <= 1
        assert 0 <= density_high_beta <= 1

    def test_threshold_effect(self, sample_matrix: torch.Tensor) -> None:
        """Test effect of threshold parameter on density computation."""
        # Low threshold should include more elements
        density_low_thresh = compute_density(sample_matrix, threshold=1e-5)

        # High threshold should exclude more elements
        density_high_thresh = compute_density(sample_matrix, threshold=0.5)

        assert (
            density_low_thresh >= density_high_thresh
        )  # Higher threshold should give lower or equal density
        assert 0 <= density_low_thresh <= 1
        assert 0 <= density_high_thresh <= 1

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        W = torch.randn(4, 4)

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_density(x)

        check_gradients(func, [W])


class TestComputeLogCorrelation:
    """Test suite for log correlation computation."""

    def test_perfect_correlation(self) -> None:
        """Test correlation computation with perfectly correlated data."""
        x = torch.tensor([1.0, 2.0, 4.0, 8.0])
        y = torch.tensor([2.0, 4.0, 8.0, 16.0])  # y = 2x
        corr = compute_log_correlation(x, y)
        assert torch.abs(corr - 1.0) < 1e-6

    def test_perfect_anticorrelation(self) -> None:
        """Test correlation computation with perfectly anti-correlated data."""
        x = torch.tensor([1.0, 2.0, 4.0, 8.0])
        y = torch.tensor([8.0, 4.0, 2.0, 1.0])  # y = 8/x
        corr = compute_log_correlation(x, y)
        assert torch.abs(corr + 1.0) < 1e-6

    def test_no_correlation(self) -> None:
        """Test correlation computation with uncorrelated data."""
        # Using fixed random seed for reproducibility
        torch.manual_seed(42)
        x = torch.exp(torch.randn(1000))
        y = torch.exp(torch.randn(1000))
        corr = compute_log_correlation(x, y)
        assert torch.abs(corr) < 0.1  # Should be close to 0

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        # Different shapes
        with pytest.raises(ValueError):
            compute_log_correlation(torch.randn(3), torch.randn(4))

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_log_correlation(np.random.randn(3), torch.randn(3))

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        # Use values > 1.0 to avoid log(x) near -infinity
        x = torch.exp(torch.randn(10).clamp(-1.0, 1.0))  # Values in [~0.37, ~2.72]
        y = torch.exp(torch.randn(10).clamp(-1.0, 1.0))

        def func(x_: torch.Tensor, y_: torch.Tensor) -> torch.Tensor:
            return compute_log_correlation(x_, y_)

        check_gradients(func, [x, y])


class TestComputeIOMatrix:
    """Test suite for IO matrix computation."""

    def test_basic_aggregation(
        self, sample_matrix: torch.Tensor, sample_group_matrix: torch.Tensor
    ) -> None:
        """Test basic IO matrix aggregation."""
        io_matrix = compute_io_matrix(sample_matrix, sample_group_matrix)

        # Manual computation for verification
        expected = (
            torch.tensor(
                [
                    [6.0, 9.0],  # Group 1 -> Group 1, Group 1 -> Group 2
                    [15.0, 0.0],  # Group 2 -> Group 1, Group 2 -> Group 2
                ]
            )
            + 1e-8
        )

        assert torch.allclose(io_matrix, expected)

    def test_gradient(self, controlled_matrix: torch.Tensor) -> None:
        """Test gradient computation using finite differences."""
        W = controlled_matrix  # Use controlled range matrix
        group_matrix = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_io_matrix(x, group_matrix).sum()

        check_gradients(func, [W])

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        # Non-square matrix
        with pytest.raises(ValueError):
            compute_io_matrix(torch.randn(3, 4), torch.randn(2, 3))

        # Invalid group matrix shape
        with pytest.raises(ValueError):
            compute_io_matrix(torch.randn(3, 3), torch.randn(2, 4))

        # Invalid group assignments
        with pytest.raises(ValueError):
            compute_io_matrix(
                torch.randn(3, 3), torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)
            )

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_io_matrix(np.random.randn(3, 3), torch.randn(2, 3))

    def test_random_directions(self) -> None:
        """Test gradient computation using random directions."""
        W = torch.randn(4, 4)
        group_matrix = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)

        # Generate random direction vector (spherical distribution)
        torch.manual_seed(42)
        d = torch.randn(4, 4)
        d = d / torch.norm(d)

        def func(eps: float) -> torch.Tensor:
            """Function to compute value in direction d."""
            return compute_io_matrix(W + eps * d, group_matrix).sum()

        # Compute directional derivative using finite differences
        eps = 1e-6
        fd_grad = (func(eps) - func(-eps)) / (2 * eps)

        # Compute analytical directional derivative
        W = W.clone().detach().requires_grad_(True)
        y = compute_io_matrix(W, group_matrix).sum()
        y.backward()
        analytical_grad = torch.sum(W.grad * d)

        # Compare directional derivatives with appropriate tolerances
        abs_diff = torch.abs(analytical_grad - fd_grad)
        abs_fd = torch.abs(fd_grad)

        if abs_fd > 1e-3:  # Use relative error for significant gradients
            rel_error = abs_diff / abs_fd
            assert rel_error < 1.0, f"Relative error too large: {rel_error.item()}"
        else:  # Use absolute error for small gradients
            assert abs_diff < 1e-3, f"Absolute error too large: {abs_diff.item()}"


class TestComputeCCDF:
    """Test suite for CCDF computation."""

    def test_step_function(self) -> None:
        """Test CCDF computation with high beta (approaching step function)."""
        values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        thresholds = torch.tensor([2.5, 3.5])
        beta_ccdf = 10.0  # High beta for sharp sigmoid

        ccdf = compute_ccdf(values, thresholds, beta_ccdf)

        # Expected: For threshold 2.5, values [3,4,5] are above (3/5)
        #           For threshold 3.5, values [4,5] are above (2/5)
        expected = torch.tensor([0.6, 0.4]) + 1e-8
        assert torch.allclose(ccdf, expected, rtol=0.1)

    def test_uniform_distribution(self) -> None:
        """Test CCDF computation with uniformly spaced values."""
        values = torch.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
        thresholds = torch.tensor([0.25, 0.5, 0.75])
        beta_ccdf = 20.0  # Sharp thresholding

        ccdf = compute_ccdf(values, thresholds, beta_ccdf)

        # Expected: Roughly 0.75, 0.5, 0.25 for uniform distribution
        # Use slightly larger tolerance for tail values where relative differences matter more
        expected = torch.tensor([0.75, 0.5, 0.25]) + 1e-8
        assert torch.allclose(
            ccdf, expected, rtol=0.15
        )  # Increased tolerance for discrete approximation

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        # 2D values tensor
        with pytest.raises(ValueError):
            compute_ccdf(torch.randn(3, 3), torch.tensor([1.0]), 1.0)

        # 2D thresholds tensor
        with pytest.raises(ValueError):
            compute_ccdf(torch.tensor([1.0]), torch.randn(3, 3), 1.0)

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_ccdf(np.array([1.0]), torch.tensor([1.0]), 1.0)

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        values = torch.randn(5).abs()  # Use positive values
        thresholds = torch.linspace(0.1, 0.9, 3)
        beta_ccdf = 1.0

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_ccdf(x, thresholds, beta_ccdf).sum()

        check_gradients(func, [values])


class TestComputeSmoothnesspenalty:
    """Test suite for smoothness penalty computation."""

    def test_linear_ccdf(self) -> None:
        """Test smoothness penalty with linear log-CCDF (should be near zero)."""
        # Create log-linear CCDF values
        thresholds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        log_ccdf = -0.5 * torch.log(thresholds)  # Linear in log-space
        ccdf_values = torch.exp(log_ccdf)

        penalty = compute_smoothness_penalty(ccdf_values, thresholds)
        assert penalty < 1e-6

    def test_nonlinear_ccdf(self) -> None:
        """Test smoothness penalty with non-linear log-CCDF."""
        # Create CCDFs with different degrees of non-linearity
        thresholds = torch.tensor([1.0, 2.0, 3.0, 4.0])
        a1 = 0.5  # Base non-linearity parameter
        a2 = 2 * a1  # Double the non-linearity

        # Create quadratic log-CCDFs: log(CCDF) = -a * (log(t))^2
        log_ccdf1 = -a1 * torch.log(thresholds) * torch.log(thresholds)
        log_ccdf2 = -a2 * torch.log(thresholds) * torch.log(thresholds)
        ccdf1 = torch.exp(log_ccdf1)
        ccdf2 = torch.exp(log_ccdf2)

        # Compute penalties
        penalty1 = compute_smoothness_penalty(ccdf1, thresholds)
        penalty2 = compute_smoothness_penalty(ccdf2, thresholds)

        # Test properties of the smoothness penalty:
        # 1. Penalty should be positive
        assert penalty1 > 0, "Smoothness penalty should be positive"

        # 2. More non-linear CCDF should have larger penalty
        assert penalty2 > penalty1, "More curved CCDF should have larger penalty"

        # 3. Doubling non-linearity should approximately quadruple penalty
        # (allowing some deviation due to discrete approximation)
        penalty_ratio = penalty2 / penalty1
        assert (
            torch.abs(penalty_ratio - 4.0) < 1.0
        ), f"Expected penalty ratio ~4.0 for doubled non-linearity, got {penalty_ratio.item():.2f}"

    def test_invalid_input(self) -> None:
        """Test error handling for invalid inputs."""
        # Different shapes
        with pytest.raises(ValueError):
            compute_smoothness_penalty(torch.randn(3), torch.randn(4))

        # 2D tensors
        with pytest.raises(ValueError):
            compute_smoothness_penalty(torch.randn(3, 3), torch.randn(3, 3))

        # Non-tensor input
        with pytest.raises(TypeError):
            compute_smoothness_penalty(np.array([1.0]), torch.tensor([1.0]))

    def test_gradient(self) -> None:
        """Test gradient computation using finite differences."""
        thresholds = torch.linspace(1.0, 4.0, 5)
        ccdf_values = torch.exp(-0.5 * thresholds) + 0.1  # Add offset to avoid zero

        def func(x: torch.Tensor) -> torch.Tensor:
            return compute_smoothness_penalty(x, thresholds)

        check_gradients(func, [ccdf_values])
