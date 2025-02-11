"""Data loading and processing utilities for network generation."""

import pickle
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch


class IOData:
    """Handler for Input-Output table data."""

    def __init__(self, io_path: Path, country: str, year: int):
        """
        Initialize IOData handler.

        Args:
            io_path: Path to IO tables directory
            country: Country code (e.g., 'FRA')
            year: Year of the IO table
        """
        self.path = io_path
        self.country = country
        self.year = year
        self._data = None
        self._tensor = None
        self._load_data()

    def _load_data(self) -> None:
        """Load IO table data from pickle file."""
        io_path = (self.path / self.country).with_suffix(".pkl")

        if not io_path.exists():
            raise FileNotFoundError(f"IO table not found at {io_path}")

        with open(io_path, "rb") as f:
            self._data = pickle.load(f)

        if self.year not in self._data:
            raise KeyError(f"Year {self.year} not found in IO table")

        self._tensor = torch.tensor(self._data[self.year])

    @property
    def tensor(self) -> torch.Tensor:
        """Get IO table as PyTorch tensor."""
        return self._tensor

    @property
    def numpy(self) -> np.ndarray:
        """Get IO table as numpy array."""
        return self._tensor.numpy()


class FirmDistribution:
    """Handler for firm distribution across industries."""

    def __init__(self, num_firms: int, io_data: Union[torch.Tensor, np.ndarray]):
        """
        Initialize firm distribution.

        Args:
            num_firms: Total number of firms
            io_data: Input-output table data
        """
        self.num_firms = num_firms
        if isinstance(io_data, torch.Tensor):
            io_data = io_data.numpy()

        self.industry_sizes = self._compute_industry_sizes(io_data)
        self.distribution = self._compute_distribution()

    def _compute_industry_sizes(self, io_data: np.ndarray) -> np.ndarray:
        """Compute number of firms per industry based on IO table proportions."""
        return np.round(self.num_firms * (np.sum(io_data, axis=1) / np.sum(io_data))).astype(int)

    def _compute_distribution(self) -> Dict[int, np.ndarray]:
        """Compute firm indices for each industry."""
        firm_idx = np.arange(self.num_firms)
        cumsum = self.industry_sizes.cumsum()

        distribution = {0: firm_idx[: cumsum[0]]}
        for i in range(1, len(self.industry_sizes)):
            distribution[i] = firm_idx[cumsum[i - 1] : cumsum[i]]

        return distribution

    def get_industry_matrix(self) -> torch.Tensor:
        """
        Create industry membership matrix.

        Returns:
            Matrix S where S[i,j] = 1 if firm j belongs to industry i
        """
        num_industries = len(self.distribution)
        S = torch.zeros((num_industries, self.num_firms), dtype=torch.float32)

        for i, firms in self.distribution.items():
            S[i, firms] = 1

        return S

    @property
    def indices(self) -> Dict[int, np.ndarray]:
        """Get firm indices for each industry."""
        return self.distribution


def compute_aggregated_io(
    weighted_matrix: torch.Tensor,
    firm_distribution: Union[FirmDistribution, Dict[int, np.ndarray]],
) -> torch.Tensor:
    """
    Compute aggregated IO table from weighted matrix.

    Args:
        weighted_matrix: Firm-level weight matrix
        firm_distribution: Mapping of industry indices to firm indices
                         or FirmDistribution instance

    Returns:
        Aggregated industry-level IO table
    """
    if isinstance(firm_distribution, FirmDistribution):
        industry_matrix = firm_distribution.get_industry_matrix()
    else:
        num_industries = len(firm_distribution)
        num_firms = weighted_matrix.shape[0]
        # Create industry matrix with same dtype as weighted_matrix
        industry_matrix = torch.zeros((num_industries, num_firms), dtype=weighted_matrix.dtype)
        for i, firms in firm_distribution.items():
            industry_matrix[i, firms] = 1

    # Ensure industry_matrix is on the same device as weighted_matrix
    industry_matrix = industry_matrix.to(weighted_matrix.device)

    return torch.matmul(torch.matmul(industry_matrix, weighted_matrix), industry_matrix.T)
