"""Network generator model implementation.

This module implements the core network generation model using matrix factorization
with learnable parameters.
"""

import logging
from typing import TypeVar

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

    def __init__(self, config: NetworkConfig) -> None:
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
        self.alpha = nn.Parameter(torch.randn(self.M) / torch.sqrt(torch.tensor(self.M)))

        # Initialize U and V matrices for each component
        # Using Xavier/Glorot initialization for better gradient flow
        self.U = nn.Parameter(torch.randn(self.M, self.N) / torch.sqrt(torch.tensor(self.N)))
        self.V = nn.Parameter(torch.randn(self.M, self.N) / torch.sqrt(torch.tensor(self.N)))

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
    def from_pretrained(cls: type[T], state_dict_path: str) -> T:
        """Create model from pretrained state dict.

        Args:
            state_dict_path: Path to saved state dict

        Returns:
            Initialized model with loaded weights

        Raises:
            ValueError: If state dict is incompatible
        """
        state_dict = torch.load(state_dict_path)

        # Extract N and M from state dict shapes
        M = state_dict["alpha"].shape[0]
        N = state_dict["U"].shape[1]

        # Create config with minimal information needed for initialization
        config = {"N": N, "M": M}

        # Create model and load state dict
        model = cls(config)
        model.load_state_dict(state_dict)

        return model

    def get_network_weights(self) -> torch.Tensor:
        """Get actual network weights W = exp(M).

        Returns:
            torch.Tensor: Weight matrix W of shape (N, N)
        """
        return torch.exp(self.compute_log_weights())
