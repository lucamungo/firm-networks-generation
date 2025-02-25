"""Network generator model implementation.

This module implements the core network generation model using matrix factorization
with learnable parameters.
"""

import logging
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
import torch.nn as nn

from .parse_config import NetworkConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="NetworkGenerator")


class NetworkGenerator(nn.Module):
    """Network generator using matrix factorization.

    This model generates a directed, weighted network using a low-rank matrix
    factorization approach: M_ij = sum_m alpha_m * u_m(i) * v_m(j).
    The actual network weights are obtained as W_ij = exp(M_ij).
    """

    def __init__(
        self, config: NetworkConfig, normalize: bool = True, loc: float = 0, std: float = 1.0
    ) -> None:
        """Initialize network generator.

        Args:
            config: Configuration dictionary containing N (number of nodes) and
                   M (number of components)
        """
        super().__init__()

        self.N = config["N"]
        self.M = config["M"]

        # Initialize learnable parameters
        # We use nn.Parameter to automatically track gradients
        # Using double precision for gradient checking
        self.alpha = nn.Parameter(
            torch.randn(self.M, dtype=torch.float64) / torch.sqrt(torch.tensor(self.M))
        )

        # Initialize U and V matrices for each component

        self.U = nn.Parameter(torch.randn(self.M, self.N, dtype=torch.float64) * std + loc)
        self.V = nn.Parameter(torch.randn(self.M, self.N, dtype=torch.float64) * std + loc)

        if normalize:
            # Xavier/Glorot initialization for better gradient flow
            self.U = nn.Parameter(self.U / torch.norm(self.U, dim=1).unsqueeze(1))
            self.V = nn.Parameter(self.V / torch.norm(self.V, dim=1).unsqueeze(1))

        logger.info(f"Initialized NetworkGenerator with N={self.N}, M={self.M}")

    def compute_log_weights(self) -> torch.Tensor:
        """Compute log-weight matrix M.

        Returns:
            torch.Tensor: Log-weight matrix M of shape (N, N)
        """
        # Compute M_ij = sum_m alpha_m * u_m(i) * v_m(j)
        # We can do this efficiently using matrix operations

        # First multiply alpha with U to get (M, N)
        alpha_U = self.alpha.unsqueeze(1) * self.U

        # Then compute the matrix product with V.T to get (N, N)
        M = torch.matmul(alpha_U.T, self.V)

        return M

    def forward(self) -> torch.Tensor:
        """Generate network log-weight matrix.

        Returns:
            torch.Tensor: Log-weight matrix M of shape (N, N)
        """
        return self.compute_log_weights()

    @classmethod
    def from_pretrained(cls: type[T], state_dict_path: str | Path) -> T:
        """Create model from pretrained state dict.

        Args:
            state_dict_path: Path to saved state dict

        Returns:
            Initialized model with loaded weights

        Raises:
            ValueError: If state dict is incompatible
        """
        checkpoint = torch.load(state_dict_path)
        state_dict = checkpoint["model_state_dict"]
        config = checkpoint["config"]

        # Create model from config
        model = cls(config)
        model.load_state_dict(state_dict)
        return model

    def get_network_weights(self) -> torch.Tensor:
        """Get actual network weights W = exp(M).

        Returns:
            torch.Tensor: Weight matrix W of shape (N, N)
        """
        return torch.exp(self.compute_log_weights())


class ModularNetworkGenerator(nn.Module):
    """Network generator with separated parameter groups for different properties."""

    def __init__(self, config):
        super().__init__()
        self.N = config["N"]

        # Density control parameters
        self.density_factor = nn.Parameter(torch.tensor([0.0]))

        # Degree distribution parameters
        M_degree = config.get("M_degree", 5)
        self.degree_alpha = nn.Parameter(torch.randn(M_degree) / np.sqrt(M_degree))
        self.degree_U = nn.Parameter(torch.randn(M_degree, self.N))
        self.degree_V = nn.Parameter(torch.randn(M_degree, self.N))

        # IO structure parameters
        M_io = config.get("M_io", 5)
        self.io_alpha = nn.Parameter(torch.randn(M_io) / np.sqrt(M_io))
        self.io_U = nn.Parameter(torch.randn(M_io, self.N))
        self.io_V = nn.Parameter(torch.randn(M_io, self.N))

        # Strength distribution parameters
        M_strength = config.get("M_strength", 5)
        self.strength_alpha = nn.Parameter(torch.randn(M_strength) / np.sqrt(M_strength))
        self.strength_U = nn.Parameter(torch.randn(M_strength, self.N))
        self.strength_V = nn.Parameter(torch.randn(M_strength, self.N))

    def compute_log_weights(self):
        # Base matrix with uniform density
        base_density = torch.zeros((self.N, self.N))
        density_mask = 1.0 - torch.eye(self.N)  # Remove diagonal
        density_component = torch.sigmoid(self.density_factor) * density_mask

        # Degree distribution component
        degree_alpha_U = self.degree_alpha.unsqueeze(1) * self.degree_U
        degree_component = torch.matmul(degree_alpha_U.T, self.degree_V)

        # IO structure component
        io_alpha_U = self.io_alpha.unsqueeze(1) * self.io_U
        io_component = torch.matmul(io_alpha_U.T, self.io_V)

        # Strength distribution component
        strength_alpha_U = self.strength_alpha.unsqueeze(1) * self.strength_U
        strength_component = torch.matmul(strength_alpha_U.T, self.strength_V)

        # Combine components
        M = density_component + degree_component + io_component + strength_component

        return M
