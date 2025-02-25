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
from scipy import stats

from .parse_config import NetworkConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="NetworkGenerator")


def initialize_from_target_power_laws(config):
    """Generate initialization values for NetworkGenerator with correct density approach.

    Args:
        config: Configuration dictionary with target network properties

    Returns:
        tuple: (alpha, U, V) as numpy arrays for initializing NetworkGenerator
    """
    import numpy as np

    N = config["N"]
    M = config["M"]
    target_density = config["density_target"]  # Percentage
    beta_degree = config.get("beta_degree", 5.0)
    threshold_degree = config.get("threshold_degree", 1e-5)

    # Extract target exponents and correlations
    in_hill = config["hill_exponent_targets"]["in_degree"]
    out_hill = config["hill_exponent_targets"]["out_degree"]
    in_strength_hill = config["hill_exponent_targets"]["in_strength"]
    out_strength_hill = config["hill_exponent_targets"]["out_strength"]
    target_corr = config["correlation_targets"]["log_in_degree_out_degree"]
    target_str_corr = config["correlation_targets"]["log_in_strength_out_strength"]

    print(
        f"Target properties: Density={target_density:.4f}%, Hill exponents: in={in_hill:.2f}, out={out_hill:.2f}"
    )

    # 1. Generate correlated log-normal degree sequences
    sigma_in = np.sqrt(2 / abs(in_hill))
    sigma_out = np.sqrt(2 / abs(out_hill))

    # Create covariance matrix for degree generation
    cov = np.array(
        [
            [sigma_in**2, target_corr * sigma_in * sigma_out],
            [target_corr * sigma_in * sigma_out, sigma_out**2],
        ]
    )

    # Generate correlated log degrees
    log_degrees = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=N)
    raw_in_degrees = np.exp(log_degrees[:, 0])
    raw_out_degrees = np.exp(log_degrees[:, 1])

    # Normalize to maintain expected values
    in_degrees = raw_in_degrees / np.mean(raw_in_degrees)  # Mean = 1
    out_degrees = raw_out_degrees / np.mean(raw_out_degrees)  # Mean = 1

    # Calculate binary density needed based on expected nodes and edges
    # Target is in percentage, convert to proportion
    target_density_prop = target_density / 100.0
    expected_edges = N * N * target_density_prop

    # Estimate how many edges we need in the binary adjacency matrix
    # This maps directly to the expected edges in the weight matrix
    binary_density_needed = expected_edges / (N * N)

    # Create affinity matrix S as outer product of in and out degrees
    S = np.outer(in_degrees, out_degrees)

    # Normalize S so average equals our binary density target
    S = S / np.mean(S) * binary_density_needed

    # Ensure values are between 0 and 1 (probabilities)
    S = np.minimum(S, 1.0)

    # Remove diagonal (no self-loops)
    np.fill_diagonal(S, 0)

    # Sample active links based on these probabilities
    adjacency = (np.random.random((N, N)) < S).astype(float)

    # 3. Generate node strengths using similar approach for correlation
    sigma_in_str = np.sqrt(2 / abs(in_strength_hill))
    sigma_out_str = np.sqrt(2 / abs(out_strength_hill))

    # Create correlation matrix for in/out strengths
    cov_str = np.array(
        [
            [sigma_in_str**2, target_str_corr * sigma_in_str * sigma_out_str],
            [target_str_corr * sigma_in_str * sigma_out_str, sigma_out_str**2],
        ]
    )

    # Generate correlated log strengths
    log_strengths = np.random.multivariate_normal(mean=[0, 0], cov=cov_str, size=N)
    in_strengths = np.exp(log_strengths[:, 0])
    out_strengths = np.exp(log_strengths[:, 1])

    # Normalize strengths for reasonable values
    in_strengths = in_strengths / np.mean(in_strengths) * 10  # Mean = 10
    out_strengths = out_strengths / np.mean(out_strengths) * 10  # Mean = 10

    # 4. Create weight matrix - distribute strengths along existing edges
    weight_matrix = np.zeros((N, N))

    # Efficient strength distribution
    for i in range(N):
        out_neighbors = np.where(adjacency[i] > 0)[0]
        if len(out_neighbors) > 0:
            # Distribute out-strength proportionally to all out neighbors
            weight_matrix[i, out_neighbors] = out_strengths[i] / len(out_neighbors)

    # One iteration of column normalization to better match in-strengths
    for j in range(N):
        in_neighbors = np.where(adjacency[:, j] > 0)[0]
        if len(in_neighbors) > 0:
            # Scale the incoming weights to match the target in-strength
            current_in_strength = weight_matrix[:, j].sum()
            if current_in_strength > 0:
                weight_matrix[:, j] = weight_matrix[:, j] * (in_strengths[j] / current_in_strength)

    # 5. Take log and perform SVD
    eps = 1e-20  # Small value to avoid log(0)
    log_weights = np.log(weight_matrix + eps)

    # Check density of weight matrix using the sigmoid approach from stats.py
    soft_adjacency = 1.0 / (1.0 + np.exp(-beta_degree * (log_weights - threshold_degree)))
    realized_density = np.mean(soft_adjacency) * 100
    print(
        f"Initial weight matrix density: {realized_density:.4f}%; {np.mean(weight_matrix > 0):.4f}"
    )

    # Double-check density after exp(log_weights)
    reconstructed_density = np.mean(soft_adjacency > 1e-16) * 100
    print(f"Reconstructed weights density: {reconstructed_density:.4f}%")

    # Perform SVD efficiently
    from scipy.sparse.linalg import svds

    # Use full SVD for small matrices, svds for large ones
    if N <= 1000:
        U, S, Vt = np.linalg.svd(log_weights, full_matrices=False)
        # Truncate to M components
        U = U[:, :M]
        S = S[:M]
        Vt = Vt[:M, :]
    else:
        # For large matrices, use svds which is more efficient
        U, S, Vt = svds(log_weights, k=min(M, N - 1))
        # Sort by singular values (svds doesn't guarantee order)
        idx = np.argsort(S)[::-1]
        U = U[:, idx]
        S = S[idx]
        Vt = Vt[idx, :]

    # Initialize the model parameters
    alpha_init = np.ones(M)
    U_init = np.zeros((M, N))
    V_init = np.zeros((M, N))

    # Fill with available SVD components
    M_actual = min(M, len(S))
    for i in range(M_actual):
        scale = np.sqrt(S[i])
        U_init[i] = U[:, i] * scale
        V_init[i] = Vt[i, :] * scale
        alpha_init[i] = 1.0

    # Set any remaining components to small random values
    if M > M_actual:
        for i in range(M_actual, M):
            alpha_init[i] = 0.001
            U_init[i] = np.random.randn(N) * 0.01
            V_init[i] = np.random.randn(N) * 0.01

    # Final verification of reconstruction
    final_reconstructed = np.zeros((N, N))
    for i in range(M):
        final_reconstructed += alpha_init[i] * np.outer(U_init[i], V_init[i])

    # Check final density after SVD reconstruction
    final_weights = np.exp(final_reconstructed)
    soft_adjacency = 1 / (1 + np.exp(-beta_degree * (final_weights - threshold_degree)))
    final_density = np.mean(soft_adjacency) * 100
    print(f"Final log-weight matrix density: {final_density:.4f}%")

    # Compute reconstruction error
    error = np.mean((final_reconstructed - log_weights) ** 2)
    print(f"SVD reconstruction error: {error:.4f}")

    return alpha_init, U_init, V_init


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


class CSHNetworkGenerator(NetworkGenerator):
    """Network generator with CSH initialization."""

    def __init__(self, config: NetworkConfig) -> None:

        super().__init__(config)

        print("Initializing U, V, alpha")
        alpha_init, U_init, V_init = initialize_from_target_power_laws(config)

        # Convert to torch tensors
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float64))
        self.U = nn.Parameter(torch.tensor(U_init, dtype=torch.float64))
        self.V = nn.Parameter(torch.tensor(V_init, dtype=torch.float64))


class EnhancedNetworkGenerator(nn.Module):
    """Network generator with explicit density control, node strengths, and connection patterns."""

    def __init__(
        self, config: dict, normalize: bool = True, loc: float = 0, std: float = 1.0
    ) -> None:
        super().__init__()

        self.N = config["N"]
        self.M = config["M"]  # Components for connection patterns

        # Global density parameter - initialize based on target density
        target_density = config.get("density_target", 0.5)
        # Use logit of target density as initial value to get faster convergence
        initial_density = torch.tensor(
            np.log(target_density / (1 - target_density)), dtype=torch.float64
        )
        self.global_density = nn.Parameter(initial_density)

        # Node-specific strength parameters
        self.log_in_strength = nn.Parameter(torch.zeros(self.N, dtype=torch.float64))
        self.log_out_strength = nn.Parameter(torch.zeros(self.N, dtype=torch.float64))

        # Initialize alpha for connection pattern components
        self.alpha = nn.Parameter(
            torch.randn(self.M, dtype=torch.float64) / torch.sqrt(torch.tensor(self.M))
        )

        # Initialize U and V matrices for connection patterns
        self.U = nn.Parameter(torch.randn(self.M, self.N, dtype=torch.float64) * std + loc)
        self.V = nn.Parameter(torch.randn(self.M, self.N, dtype=torch.float64) * std + loc)

        if normalize:
            # Xavier/Glorot initialization for better gradient flow
            self.U = nn.Parameter(self.U / torch.norm(self.U, dim=1).unsqueeze(1))
            self.V = nn.Parameter(self.V / torch.norm(self.V, dim=1).unsqueeze(1))

    def compute_log_weights(self) -> torch.Tensor:
        """Compute log-weight matrix M with global density, node strengths, and patterns.

        Returns:
            torch.Tensor: Log-weight matrix M of shape (N, N)
        """
        # Apply global density scaling using sigmoid
        density_factor = torch.sigmoid(self.global_density)

        # Compute the base pattern using alpha, U, and V
        alpha_U = self.alpha.unsqueeze(1) * self.U
        base_pattern = torch.matmul(alpha_U.T, self.V)

        # Apply node-specific strength scaling
        out_strength_factor = self.log_out_strength.unsqueeze(1)  # Shape: (N, 1)
        in_strength_factor = self.log_in_strength.unsqueeze(0)  # Shape: (1, N)

        # Combine all factors
        # First combine the base pattern with strength factors
        M = base_pattern + out_strength_factor + in_strength_factor

        # Apply global density scaling - add log(density_factor) to shift all values
        # This scales all edge probabilities by the same factor
        M = M + torch.log(density_factor + 1e-16)

        # Zero out diagonal elements; set them to 1e-16
        M = M * (1 - torch.eye(self.N, device=M.device)) + 1e-16 * torch.eye(
            self.N, device=M.device
        )

        return M

    def forward(self) -> torch.Tensor:
        """Generate network log-weight matrix.

        Returns:
            torch.Tensor: Log-weight matrix M of shape (N, N)
        """
        return self.compute_log_weights()

    def get_network_weights(self) -> torch.Tensor:
        """Get actual network weights W = exp(M).

        Returns:
            torch.Tensor: Weight matrix W of shape (N, N)
        """
        return torch.exp(self.compute_log_weights())

    def freeze_density(self):
        """Freeze the global density parameter."""
        self.global_density.requires_grad = False

    def unfreeze_density(self):
        """Unfreeze the global density parameter."""
        self.global_density.requires_grad = True

    def get_current_density(self) -> float:
        """Get the current density parameter value (as probability)."""
        return torch.sigmoid(self.global_density).item()


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


class DisentangledNetworkGenerator(nn.Module):
    """Network generator with separated adjacency and strength components.

    This model explicitly separates the binary existence of connections (adjacency)
    from their weights (strengths), allowing for better control of network sparsity
    while preserving other statistical properties.
    """

    def __init__(self, config):
        super().__init__()

        self.N = config["N"]
        self.M_adj = config.get("M_adj", config["M"])  # Components for adjacency
        self.M_str = config.get("M_str", config["M"])  # Components for strength

        # Parameters for adjacency matrix (binary connections)
        self.adj_alpha = nn.Parameter(
            torch.randn(self.M_adj, dtype=torch.float64) / np.sqrt(self.M_adj)
        )
        self.adj_U = nn.Parameter(torch.randn(self.M_adj, self.N, dtype=torch.float64))
        self.adj_V = nn.Parameter(torch.randn(self.M_adj, self.N, dtype=torch.float64))

        # Parameters for strength matrix (connection weights)
        self.str_alpha = nn.Parameter(
            torch.randn(self.M_str, dtype=torch.float64) / np.sqrt(self.M_str)
        )
        self.str_U = nn.Parameter(torch.randn(self.M_str, self.N, dtype=torch.float64))
        self.str_V = nn.Parameter(torch.randn(self.M_str, self.N, dtype=torch.float64))

        # Global sparsity control - initialized based on target density
        target_density = config.get("density_target", 5.0) / 100.0  # Convert % to proportion
        initial_logit = np.log(target_density / (1 - target_density))
        self.sparsity = nn.Parameter(torch.tensor([initial_logit], dtype=torch.float64))

        # Initialize properly if requested
        if config.get("initialize_from_target", True):
            self.initialize_from_target(config)

    def initialize_from_target(self, config):
        """Initialize model parameters from target properties."""
        # Generate a network with target properties
        N = self.N
        target_density = config.get("density_target", 5.0) / 100.0

        # 1. Generate degree sequences from target Hill exponents
        in_hill = config["hill_exponent_targets"]["in_degree"]
        out_hill = config["hill_exponent_targets"]["out_degree"]
        target_corr = config["correlation_targets"]["log_in_degree_out_degree"]

        # Generate correlated log-normal degree sequences
        sigma_in = np.sqrt(2 / abs(in_hill))
        sigma_out = np.sqrt(2 / abs(out_hill))

        # Create covariance matrix for degree generation
        cov = np.array(
            [
                [sigma_in**2, target_corr * sigma_in * sigma_out],
                [target_corr * sigma_in * sigma_out, sigma_out**2],
            ]
        )

        # Generate correlated log degrees
        log_degrees = np.random.multivariate_normal(mean=[0, 0], cov=cov, size=N)
        in_degrees = np.exp(log_degrees[:, 0])
        out_degrees = np.exp(log_degrees[:, 1])

        # 2. Create adjacency matrix using outer product approach
        S = np.outer(out_degrees, in_degrees)  # Connection probability matrix
        S = S / np.mean(S) * target_density  # Scale to match target density
        S = np.minimum(S, 1.0)  # Ensure probabilities are <= 1
        np.fill_diagonal(S, 0)  # No self-loops

        # Sample adjacency matrix
        adjacency = (np.random.random((N, N)) < S).astype(float)

        # 3. Generate strengths with target Hill exponents
        in_strength_hill = config["hill_exponent_targets"]["in_strength"]
        out_strength_hill = config["hill_exponent_targets"]["out_strength"]
        target_str_corr = config["correlation_targets"]["log_in_strength_out_strength"]

        sigma_in_str = np.sqrt(2 / abs(in_strength_hill))
        sigma_out_str = np.sqrt(2 / abs(out_strength_hill))

        # Generate correlated strengths
        cov_str = np.array(
            [
                [sigma_in_str**2, target_str_corr * sigma_in_str * sigma_out_str],
                [target_str_corr * sigma_in_str * sigma_out_str, sigma_out_str**2],
            ]
        )

        log_strengths = np.random.multivariate_normal(mean=[0, 0], cov=cov_str, size=N)
        in_strengths = np.exp(log_strengths[:, 0])
        out_strengths = np.exp(log_strengths[:, 1])

        # Create strength matrix for existing edges
        strength_matrix = np.zeros((N, N))
        for i in range(N):
            out_neighbors = np.where(adjacency[i] > 0)[0]
            if len(out_neighbors) > 0:
                strength_matrix[i, out_neighbors] = out_strengths[i] / len(out_neighbors)

        # 4. Convert to log-strength for existing edges
        # We only care about strength values where adjacency is 1
        log_strength_matrix = np.zeros((N, N))
        mask = adjacency > 0
        log_strength_matrix[mask] = np.log(strength_matrix[mask] + 1e-16)

        # 5. Compute logits for adjacency matrix
        # logit(p) = log(p/(1-p))
        epsilon = 1e-16
        S_safe = np.clip(S, epsilon, 1 - epsilon)  # Avoid log(0) and log(inf)
        logit_matrix = np.log(S_safe / (1 - S_safe))

        # 6. Factorize matrices using SVD
        # Adjacency logits
        U_adj, S_adj, Vt_adj = np.linalg.svd(logit_matrix, full_matrices=False)

        # Strength matrix
        U_str, S_str, Vt_str = np.linalg.svd(log_strength_matrix, full_matrices=False)

        # 7. Initialize parameters from SVD components
        M_adj_actual = min(self.M_adj, len(S_adj))
        for i in range(M_adj_actual):
            scale = np.sqrt(S_adj[i])
            self.adj_U.data[i] = torch.tensor(U_adj[:, i] * scale, dtype=torch.float64)
            self.adj_V.data[i] = torch.tensor(Vt_adj[i, :] * scale, dtype=torch.float64)
            self.adj_alpha.data[i] = torch.tensor(1.0, dtype=torch.float64)

        M_str_actual = min(self.M_str, len(S_str))
        for i in range(M_str_actual):
            scale = np.sqrt(S_str[i])
            self.str_U.data[i] = torch.tensor(U_str[:, i] * scale, dtype=torch.float64)
            self.str_V.data[i] = torch.tensor(Vt_str[i, :] * scale, dtype=torch.float64)
            self.str_alpha.data[i] = torch.tensor(1.0, dtype=torch.float64)

        # 8. Compute global sparsity offset
        # This adjusts for any discrepancy in the approximation
        with torch.no_grad():
            approx_adjacency = self.compute_adjacency()
            mean_prob = approx_adjacency.mean().item()
            target_prob = adjacency.mean()
            adjustment = np.log(target_prob / (1 - target_prob)) - np.log(
                mean_prob / (1 - mean_prob)
            )
            self.sparsity.data = torch.tensor([adjustment], dtype=torch.float64)

    def compute_adjacency_logits(self):
        """Compute adjacency logits from factorization."""
        logits = torch.zeros((self.N, self.N), dtype=torch.float64, device=self.adj_alpha.device)
        for i in range(self.M_adj):
            logits += self.adj_alpha[i] * torch.outer(self.adj_U[i], self.adj_V[i])

        # Add global sparsity control
        logits += self.sparsity

        # Zero diagonal to prevent self-loops
        logits = logits * (1 - torch.eye(self.N, device=logits.device))

        return logits

    def compute_adjacency(self):
        """Compute adjacency matrix using sigmoid."""
        return torch.sigmoid(self.compute_adjacency_logits())

    def compute_strengths(self):
        """Compute log-strength matrix."""
        log_strengths = torch.zeros(
            (self.N, self.N), dtype=torch.float64, device=self.str_alpha.device
        )
        for i in range(self.M_str):
            log_strengths += self.str_alpha[i] * torch.outer(self.str_U[i], self.str_V[i])

        return log_strengths

    def forward(self):
        """Generate network log-weight matrix.

        Returns:
            Log-weight matrix where exp(log_weights) gives the final network weights.
        """
        # Compute adjacency and strength matrices
        adj = self.compute_adjacency()
        log_strengths = self.compute_strengths()

        # Combine them: log(adj * exp(log_strengths))
        # For numerical stability: log(adj) + log_strengths = log(adj * exp(log_strengths))
        # But since adj can be 0, we use a safe formulation
        eps = 1e-16
        log_weights = torch.log(adj * torch.exp(log_strengths) + eps)

        return log_weights

    def get_network_weights(self):
        """Get actual network weights W."""
        # Direct computation: W = adj * exp(log_strengths)
        adj = self.compute_adjacency()
        log_strengths = self.compute_strengths()
        return adj * torch.exp(log_strengths)

    def get_adjacency_matrix(self, threshold=0.5):
        """Get the discrete adjacency matrix using threshold."""
        return (self.compute_adjacency() > threshold).float()

    def get_strength_matrix(self):
        """Get the raw strength matrix (without adjacency masking)."""
        return torch.exp(self.compute_strengths())

    def set_target_density(self, target_density_percent):
        """Set the target density (as percentage) directly.

        Args:
            target_density_percent: Target density in percentage (0-100)
        """
        target_prop = target_density_percent / 100.0
        logit_value = np.log(target_prop / (1 - target_prop))
        current_mean_logit = (self.compute_adjacency_logits() - self.sparsity).mean().item()
        new_sparsity = logit_value - current_mean_logit
        self.sparsity.data = torch.tensor([new_sparsity], dtype=torch.float64)
