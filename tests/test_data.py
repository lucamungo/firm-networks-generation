"""Tests for data loading and processing utilities."""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from network_generation.data import FirmDistribution, IOData, compute_aggregated_io


@pytest.fixture
def sample_io_data() -> dict:
    """Create sample IO table data for testing."""
    return {2020: np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)}


@pytest.fixture
def io_data_path(tmp_path: Path, sample_io_data: dict) -> Path:
    """Create temporary directory with sample IO data file."""
    io_path = tmp_path / "io_tables"
    io_path.mkdir()

    # Save sample data
    with open(io_path / "FRA.pkl", "wb") as f:
        pickle.dump(sample_io_data, f)

    return io_path


class TestIOData:
    """Test suite for IOData class."""

    def test_initialization(self, io_data_path: Path) -> None:
        """Test successful initialization of IOData."""
        io_data = IOData(io_data_path, "FRA", 2020)
        assert isinstance(io_data.tensor, torch.Tensor)
        assert io_data.tensor.shape == (3, 3)

    def test_invalid_country(self, io_data_path: Path) -> None:
        """Test initialization with non-existent country."""
        with pytest.raises(FileNotFoundError, match="IO table not found"):
            IOData(io_data_path, "XXX", 2020)

    def test_invalid_year(self, io_data_path: Path) -> None:
        """Test initialization with non-existent year."""
        with pytest.raises(KeyError, match="Year .* not found"):
            IOData(io_data_path, "FRA", 1900)

    def test_tensor_property(self, io_data_path: Path) -> None:
        """Test tensor property returns correct PyTorch tensor."""
        io_data = IOData(io_data_path, "FRA", 2020)
        expected = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32
        )
        assert torch.allclose(io_data.tensor, expected)

    def test_numpy_property(self, io_data_path: Path) -> None:
        """Test numpy property returns correct numpy array."""
        io_data = IOData(io_data_path, "FRA", 2020)
        assert np.array_equal(
            io_data.numpy, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        )


@pytest.fixture
def sample_firm_distribution() -> FirmDistribution:
    """Create sample firm distribution for testing."""
    io_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    return FirmDistribution(num_firms=10, io_data=io_data)


class TestFirmDistribution:
    """Test suite for FirmDistribution class."""

    def test_initialization(self, sample_firm_distribution: FirmDistribution) -> None:
        """Test successful initialization of FirmDistribution."""
        assert sample_firm_distribution.num_firms == 10
        assert len(sample_firm_distribution.distribution) == 2

    def test_industry_sizes(self, sample_firm_distribution: FirmDistribution) -> None:
        """Test computation of industry sizes."""
        # First industry: (1 + 2)/(1 + 2 + 3 + 4) = 0.3 * 10 = 3 firms
        # Second industry: (3 + 4)/(1 + 2 + 3 + 4) = 0.7 * 10 = 7 firms
        assert np.array_equal(sample_firm_distribution.industry_sizes, np.array([3, 7]))

    def test_distribution_indices(self, sample_firm_distribution: FirmDistribution) -> None:
        """Test firm indices distribution across industries."""
        indices = sample_firm_distribution.indices
        assert np.array_equal(indices[0], np.array([0, 1, 2]))
        assert np.array_equal(indices[1], np.array([3, 4, 5, 6, 7, 8, 9]))

    def test_industry_matrix(self, sample_firm_distribution: FirmDistribution) -> None:
        """Test creation of industry membership matrix."""
        S = sample_firm_distribution.get_industry_matrix()
        assert S.shape == (2, 10)
        assert torch.all(S.sum(dim=0) == 1)  # Each firm belongs to exactly one industry
        assert torch.sum(S[0]) == 3  # First industry has 3 firms
        assert torch.sum(S[1]) == 7  # Second industry has 7 firms


def test_compute_aggregated_io() -> None:
    """Test computation of aggregated IO table."""
    # Create sample data
    weighted_matrix = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    firm_distribution = {
        0: np.array([0, 1]),  # First industry: firms 0,1
        1: np.array([2, 3]),  # Second industry: firms 2,3
    }

    result = compute_aggregated_io(weighted_matrix, firm_distribution)

    # Expected result calculation:
    # First compute industry matrix S
    S = torch.zeros((2, 4))
    S[0, :2] = 1
    S[1, 2:] = 1

    # Expected = S @ weighted_matrix @ S.T
    expected = torch.matmul(torch.matmul(S, weighted_matrix), S.T)

    assert torch.allclose(result, expected)
    assert result.shape == (2, 2)  # Should aggregate to 2x2 matrix for 2 industries
