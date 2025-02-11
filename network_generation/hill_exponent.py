"""Power-law tail fitting utilities.

This module provides functions for fitting power-law tails to distributions
using log-log CCDF analysis. It uses a soft CCDF computation and differentiable
linear regression to maintain gradient flow.
"""

import torch

from .stats import compute_ccdf


def compute_ccdf_and_fit_tail(
    x_values: torch.Tensor,
    tail_fraction: float,
    num_points: int,
    beta_ccdf: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Input validation
    if not isinstance(x_values, torch.Tensor):
        raise TypeError("x_values must be a torch.Tensor")
    if x_values.dim() != 1:
        raise ValueError("x_values must be a 1D tensor")
    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    # Instead of sorting x_values (which is non-differentiable),
    # compute a soft minimum and maximum directly.
    x_min = torch.sum(x_values * torch.softmax(-beta_ccdf * x_values, dim=0))
    x_max = torch.sum(x_values * torch.softmax(beta_ccdf * x_values, dim=0))

    # Create thresholds in log-space in a differentiable way.
    log_min = torch.log(x_min + eps)
    log_max = torch.log(x_max + eps)
    t = torch.linspace(0, 1, num_points, device=x_values.device, dtype=x_values.dtype)
    log_thresholds = log_min + (log_max - log_min) * t
    thresholds = torch.exp(log_thresholds)

    # Compute CCDF using your existing (soft) implementation.
    ccdf_values = compute_ccdf(x_values, thresholds, beta_ccdf, eps)

    # Work in log space with soft validity mask
    log_t = log_thresholds - log_thresholds.mean()
    log_ccdf = torch.log(ccdf_values + eps)
    log_ccdf = log_ccdf - log_ccdf.mean()

    # Soft validity mask instead of hard threshold
    valid = torch.sigmoid(beta_ccdf * (ccdf_values - eps))

    # Apply mask to computations
    masked_t = log_t * valid
    masked_ccdf = log_ccdf * valid

    # Compute slope with masked values
    numerator = torch.sum(masked_t * masked_ccdf)
    denominator = torch.sum(masked_t * masked_t) + eps
    slope = numerator / denominator

    # Compute intercept
    intercept = masked_ccdf.mean() - slope * masked_t.mean()

    # Compute residuals with mask
    y_pred = slope * log_t + intercept
    ss_res = torch.mean(valid * (log_ccdf - y_pred) ** 2)

    return slope, intercept, ss_res


def hill_loss_from_fit(
    slope: torch.Tensor,
    intercept: torch.Tensor,
    ss_res: torch.Tensor,
    slope_target: float,
    lambda_line: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute loss based on slope mismatch and deviation from straight line.

    Args:
        slope: Fitted slope in log-log space
        intercept: Fitted intercept in log-log space
        ss_res: Sum of squared residuals from the fit
        slope_target: Target slope value
        lambda_line: Weight for line deviation penalty
        eps: Small value to add for numerical stability

    Returns:
        Total loss combining slope mismatch and line deviation
    """
    # Compute slope mismatch (note: slope should be negative for power laws)
    slope_loss = (slope - slope_target) ** 2

    # Add penalty for deviation from straight line
    line_loss = lambda_line * ss_res

    # Add regularization on intercept to keep it in a reasonable range
    # This helps stabilize the fit without affecting the power-law behavior
    intercept_reg = 0.01 * intercept**2

    # Combine all loss terms
    total_loss = slope_loss + line_loss + intercept_reg

    return total_loss


def compute_hill_exponent(
    values: torch.Tensor,
    tail_fraction: float = 0.3,
    beta_tail: float = 1.0,
    is_discrete: bool = False,
    temperature: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor")
    if values.dim() != 1:
        raise ValueError("values must be a 1D tensor")
    if not (0 < tail_fraction <= 1):
        raise ValueError("tail_fraction must be in (0, 1]")

    # Get tail points
    k = torch.quantile(values, tail_fraction)

    # Create soft mask
    soft_mask = torch.sigmoid((values - k) / temperature)

    # Calculate log ratios for all points
    log_ratios = torch.log(values / k)

    # Apply soft mask to log ratios
    masked_log_ratios = log_ratios * soft_mask

    # Calculate effective sample size (sum of mask)
    n_effective = soft_mask.sum()

    # Calculate hill exponent using effective sample size
    hill_exponent = n_effective * (1 / masked_log_ratios.sum())

    return hill_exponent
