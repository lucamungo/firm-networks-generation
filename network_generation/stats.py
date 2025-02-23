"""Statistical computations for network analysis.

This module provides functions for computing various network statistics,
including degrees, strengths, correlations, and aggregated matrices.
"""

import torch


def compute_degrees(
    W: torch.Tensor, beta_degree: float, dim: int = 1, eps: float = 1e-8, threshold: float = 1e-5
) -> torch.Tensor:
    """
    Compute soft degrees of a weighted adjacency matrix.

    Args:
        W: Weighted adjacency matrix of shape (N, N)
        beta_degree: Temperature parameter for softmax
        dim: Dimension to sum over (1 for in-degree, 0 for out-degree)
        threshold: Threshold for soft degree computation
        eps: Small value to add for numerical stability

    Returns:
        Tensor of shape (N,) containing the soft degrees
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError("W must be a torch.Tensor")

    if W.dim() != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    if dim not in [0, 1]:
        raise ValueError("dim must be 0 or 1")

    # Apply sigmoid with temperature
    soft_adjacency = torch.sigmoid(beta_degree * (W - threshold))

    # Sum over specified dimension and add eps for numerical stability
    return torch.sum(soft_adjacency, dim=dim) + eps


def compute_strengths(
    W: torch.Tensor,
    beta: float = 5.0,
    threshold: float = 1e-5,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute node strengths of a weighted adjacency matrix.

    Args:
        W: Weighted adjacency matrix of shape (N, N)
        dim: Dimension to sum over (1 for in-strength, 0 for out-strength)
        beta: Temperature parameter for softmax
        threshold: Threshold for soft degree computation
        eps: Small value to add for numerical stability

    Returns:
        Tensor of shape (N,) containing the node strengths
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError("W must be a torch.Tensor")

    if W.dim() != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    if dim not in [0, 1]:
        raise ValueError("dim must be 0 or 1")

    # Compute strengths with numerical stability
    # Apply sigmoid with temperature
    soft_adjacency = torch.sigmoid(beta * (W - threshold))

    soft_masked_W = soft_adjacency * W

    return torch.sum(soft_masked_W, dim=dim) + eps


def compute_log_correlation(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute correlation between log-transformed variables.

    Args:
        x: First variable
        y: Second variable
        eps: Small value to add before taking log for numerical stability

    Returns:
        Scalar tensor containing the correlation coefficient
    """
    if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
        raise TypeError("x and y must be torch.Tensor")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # Log transform with offset
    log_x = torch.log(x + eps)
    log_y = torch.log(y + eps)

    # Center the variables
    log_x = log_x - log_x.mean()
    log_y = log_y - log_y.mean()

    # Compute correlation with numerical stability
    numerator = torch.sum(log_x * log_y)
    var_x = torch.sum(log_x * log_x)
    var_y = torch.sum(log_y * log_y)
    denominator = torch.sqrt((var_x + eps) * (var_y + eps))

    return numerator / (denominator + eps)


def compute_io_matrix(
    W: torch.Tensor, group_matrix: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute aggregated input-output matrix by groups.

    Args:
        W: Weighted adjacency matrix of shape (N, N)
        group_matrix: Binary matrix of shape (G, N) where G is number of groups
                     group_matrix[i,j] = 1 if node j belongs to group i
        eps: Small value to add for numerical stability

    Returns:
        Tensor of shape (G, G) containing the aggregated IO matrix
    """
    if not isinstance(W, torch.Tensor) or not isinstance(group_matrix, torch.Tensor):
        raise TypeError("W and group_matrix must be torch.Tensor")

    if W.dim() != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    if group_matrix.dim() != 2 or group_matrix.shape[1] != W.shape[0]:
        raise ValueError("group_matrix must be a GxN matrix where N is number of nodes")

    # Check if group assignments are valid (each node belongs to exactly one group)
    node_group_sums = torch.sum(group_matrix, dim=0)
    if not torch.all(node_group_sums == 1):
        raise ValueError("Each node must belong to exactly one group")

    group_matrix = group_matrix.to(dtype=W.dtype)

    # Compute aggregated matrix: group_matrix @ W @ group_matrix.T
    return torch.matmul(torch.matmul(group_matrix, W), group_matrix.T) + eps


def compute_ccdf(
    values: torch.Tensor,
    thresholds: torch.Tensor,
    beta: float,
    eps: float = 1e-8,
    use_adaptive_beta: bool = True,
) -> torch.Tensor:
    """
    Compute complementary cumulative distribution function (CCDF) using adaptive soft thresholding.

    Args:
        values: Input values [N]
        thresholds: Points at which to evaluate CCDF [M]
        beta: Base temperature parameter for soft thresholding
        use_adaptive_beta: Use adaptive temperature based on spread of log-ratios
        eps: Small value for numerical stability

    Returns:
        CCDF values at each threshold [M]
    """
    if not isinstance(values, torch.Tensor) or not isinstance(thresholds, torch.Tensor):
        raise TypeError("values and thresholds must be torch.Tensor")

    if values.dim() != 1 or thresholds.dim() != 1:
        raise ValueError("values and thresholds must be 1D tensors")

    # Compute log values once
    log_x = torch.log(values.clamp(min=eps))  # [N]
    log_t = torch.log(thresholds.clamp(min=eps))  # [M]

    # Use broadcasting: [M, N]
    log_ratios = log_x.unsqueeze(0) - log_t.unsqueeze(1)

    # Scale beta inversely with the spread of log-ratios to maintain good gradient
    if use_adaptive_beta:
        log_ratio_std = log_ratios.std()
        if log_ratio_std > 0:
            adaptive_beta = beta / log_ratio_std.clamp(min=eps)
        else:
            adaptive_beta = beta
    else:
        adaptive_beta = beta

    # Compute indicators with adaptive temperature
    indicators = torch.sigmoid(adaptive_beta * log_ratios)

    # Average over samples
    ccdf = torch.mean(indicators, dim=1)

    return ccdf


def compute_smoothness_penalty(
    ccdf_values: torch.Tensor,
    thresholds: torch.Tensor,
    remove_mean_dt: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute smoothness penalty based on second derivative of log-CCDF.

    Args:
        ccdf_values: CCDF values at each threshold, shape (K,)
        thresholds: Points at which CCDF was evaluated, shape (K,)
        remove_mean_dt: Whether to divide by mean dt
        eps: Small value to add for numerical stability

    Returns:
        Scalar tensor containing the smoothness penalty
    """
    if not isinstance(ccdf_values, torch.Tensor) or not isinstance(thresholds, torch.Tensor):
        raise TypeError("ccdf_values and thresholds must be torch.Tensor")

    if ccdf_values.shape != thresholds.shape:
        raise ValueError("ccdf_values and thresholds must have the same shape")

    if ccdf_values.dim() != 1:
        raise ValueError("ccdf_values must be a 1D tensor")

    # Convert to log space
    log_ccdf = torch.log(ccdf_values + eps)
    log_t = torch.log(thresholds + eps)

    # Compute first differences
    dt = log_t[1:] - log_t[:-1]  # Shape: (K-1,)
    dy = log_ccdf[1:] - log_ccdf[:-1]  # Shape: (K-1,)

    # First derivative
    if remove_mean_dt:
        first_deriv = dy / (dt + eps)
    else:
        mean_dt = torch.mean(dt)
        first_deriv = dy / (dt + eps) * mean_dt

    # Second differences
    d2y = first_deriv[1:] - first_deriv[:-1]  # (K-2,)
    dt2 = dt[1:]  # (K-2,)

    if remove_mean_dt:
        second_deriv = d2y / (dt2 + eps)
    else:
        mean_dt = torch.mean(dt)
        second_deriv = d2y / (dt2 + eps) * mean_dt

    # Compute sum of squared second derivatives as smoothness penalty
    loss = torch.sum(second_deriv * second_deriv)

    # Add penalty term for second derivative > 0
    loss += torch.sum(torch.relu(-second_deriv))

    return loss


def compute_density(W: torch.Tensor, beta: float = 1.0, threshold: float = 1e-5, eps: float = 1e-8):
    """
    Compute density of a weighted adjacency matrix.
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError("W must be a torch.Tensor")

    if W.dim() != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")

    # Apply sigmoid with temperature
    soft_adjacency = torch.sigmoid(beta * (W - threshold))

    return torch.mean(soft_adjacency)
