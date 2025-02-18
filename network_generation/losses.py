"""Loss functions for network generation.

This module provides functions for computing various loss terms and combining
them into a single differentiable loss for network generation.
"""

import logging
from typing import Dict, Tuple

import torch

from .hill_exponent import compute_ccdf_and_fit_tail, hill_loss_from_fit
from .stats import (
    compute_ccdf,
    compute_degrees,
    compute_io_matrix,
    compute_log_correlation,
    compute_smoothness_penalty,
    compute_strengths,
)

logger = logging.getLogger(__name__)


def compute_correlation_loss(
    M: torch.Tensor,
    correlation_targets: Dict[str, float],
    beta_degree: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute correlation loss terms.

    Args:
        M: Log-weight matrix of shape (N, N)
        correlation_targets: Dictionary mapping correlation names to target values
        beta_degree: Temperature parameter for soft degree computation
        eps: Small value for numerical stability

    Returns:
        Tuple containing:
            - Total correlation loss
            - Dictionary of individual correlation losses
    """
    # Get weight matrix
    W = torch.exp(M)

    # Compute degrees and strengths
    in_degrees = compute_degrees(W, beta_degree, dim=0)
    out_degrees = compute_degrees(W, beta_degree, dim=1)
    in_strengths = compute_strengths(W, dim=0)
    out_strengths = compute_strengths(W, dim=1)

    # Initialize losses
    correlation_losses = {}
    total_loss = torch.tensor(0.0, device=M.device)

    # Compute each correlation and its loss
    correlations = {
        "log_in_strength_out_strength": (in_strengths, out_strengths),
        "log_in_degree_out_degree": (in_degrees, out_degrees),
        "log_out_strength_out_degree": (out_strengths, out_degrees),
    }

    for name, target in correlation_targets.items():
        if name not in correlations:
            raise ValueError(f"Unknown correlation target: {name}")

        x, y = correlations[name]
        corr = compute_log_correlation(x, y, eps)
        loss = (corr - target) ** 2
        correlation_losses[name] = loss
        total_loss = total_loss + loss

        # Log the correlation value and loss
        logger.debug(
            f"{name}: correlation = {corr.item():.3f}, "
            f"target = {target:.3f}, loss = {loss.item():.3f}"
        )

    return total_loss, correlation_losses


def compute_hill_losses(
    M: torch.Tensor,
    hill_targets: Dict[str, float],
    beta_degree: float,
    beta_tail: float,
    tail_fraction: float,
    num_points: int = 20,
    lambda_line: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute Hill exponent loss terms.

    Args:
        M: Log-weight matrix of shape (N, N)
        hill_targets: Dictionary mapping distribution names to target slopes
        beta_degree: Temperature parameter for soft degree computation
        beta_tail: Temperature parameter for CCDF computation
        tail_fraction: Fraction of data to use for tail fitting
        num_points: Number of points for CCDF evaluation
        lambda_line: Weight for line deviation penalty
        eps: Small value for numerical stability

    Returns:
        Tuple containing:
            - Total Hill exponent loss
            - Dictionary of individual Hill losses
    """
    # Get weight matrix
    W = torch.exp(M)

    # Compute degrees and strengths
    in_degrees = compute_degrees(W, beta_degree, dim=0)
    out_degrees = compute_degrees(W, beta_degree, dim=1)
    in_strengths = compute_strengths(W, dim=0)
    out_strengths = compute_strengths(W, dim=1)

    # Initialize losses
    hill_losses = {}
    total_loss = torch.tensor(0.0, device=M.device)

    # Map names to values
    distributions = {
        "in_degree": in_degrees,
        "out_degree": out_degrees,
        "in_strength": in_strengths,
        "out_strength": out_strengths,
    }

    for name, target in hill_targets.items():
        if name not in distributions:
            raise ValueError(f"Unknown Hill target: {name}")

        values = distributions[name]
        slope, intercept, ss_res = compute_ccdf_and_fit_tail(
            values,
            tail_fraction=tail_fraction,
            num_points=num_points,
            beta_ccdf=beta_tail,
            eps=eps,
        )

        loss = hill_loss_from_fit(slope, intercept, ss_res, target, lambda_line, eps)
        hill_losses[name] = loss
        total_loss = total_loss + loss

        # Log the slope and loss
        logger.debug(
            f"{name}: slope = {slope.item():.3f}, "
            f"target = {target:.3f}, residuals = {ss_res.item():.3f}, "
            f"loss = {loss.item():.3f}"
        )

    return total_loss, hill_losses


def compute_io_loss(
    M: torch.Tensor,
    group_matrix: torch.Tensor,
    io_target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute IO matrix matching loss.

    Args:
        M: Log-weight matrix of shape (N, N)
        group_matrix: Binary matrix of shape (G, N) indicating group membership
        io_target: Target IO matrix of shape (G, G)
        eps: Small value for numerical stability

    Returns:
        Loss measuring deviation from target IO matrix
    """
    # Get weight matrix
    W = torch.exp(M)

    # Cast group_matrix to same dtype as W
    group_matrix = group_matrix.to(dtype=W.dtype)

    # Compute aggregated IO matrix
    io_matrix = compute_io_matrix(W, group_matrix, eps)

    # Compute loss in log space
    log_io = torch.log(io_matrix + eps)
    log_target = torch.log(io_target + eps)
    loss = torch.mean((log_io - log_target) ** 2)

    # Log the loss
    logger.debug(f"IO matrix loss = {loss.item():.3f}")

    return loss


def compute_smoothness_loss(
    M: torch.Tensor,
    beta_degree: float,
    beta_ccdf: float,
    num_points: int = 20,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute smoothness penalties for all distributions.

    Args:
        M: Log-weight matrix of shape (N, N)
        beta_degree: Temperature parameter for soft degree computation
        beta_ccdf: Temperature parameter for CCDF computation
        num_points: Number of points for CCDF evaluation
        eps: Small value for numerical stability

    Returns:
        Tuple containing:
            - Total smoothness loss
            - Dictionary of individual smoothness losses
    """
    # Get weight matrix
    W = torch.exp(M)

    # Compute degrees and strengths
    in_degrees = compute_degrees(W, beta_degree, dim=0)
    out_degrees = compute_degrees(W, beta_degree, dim=1)
    in_strengths = compute_strengths(W, dim=0)
    out_strengths = compute_strengths(W, dim=1)

    # Initialize losses
    smoothness_losses = {}
    total_loss = torch.tensor(0.0, device=M.device)

    # Map names to values
    distributions = {
        "in_degree": in_degrees,
        "out_degree": out_degrees,
        "in_strength": in_strengths,
        "out_strength": out_strengths,
    }

    for name, values in distributions.items():
        # Generate threshold points
        # Use differentiable min and max:
        t_min = torch.min(values)
        t_max = torch.max(values)
        # Create a linearly spaced tensor in [0, 1]
        t = torch.linspace(0, 1, num_points, device=M.device, dtype=values.dtype)
        # Compute thresholds as a linear interpolation between t_min and t_max.
        thresholds = t_min + (t_max - t_min) * t

        # Compute CCDF
        ccdf_values = compute_ccdf(values, thresholds, beta_ccdf, eps)

        # Compute smoothness penalty
        loss = compute_smoothness_penalty(ccdf_values, thresholds, eps)
        smoothness_losses[name] = loss
        total_loss = total_loss + loss

        # Log the loss
        logger.debug(f"{name} smoothness loss = {loss.item():.3f}")

    return total_loss, smoothness_losses


def compute_loss(
    M: torch.Tensor,
    config: Dict,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute total loss combining all constraints.

    Args:
        M: Log-weight matrix of shape (N, N)
        config: Dictionary containing:
            - correlation_targets: Dict[str, float]
            - hill_exponent_targets: Dict[str, float]
            - io_matrix_target: torch.Tensor
            - group_matrix: torch.Tensor
            - loss_weights: Dict[str, float]
            - beta_degree: float
            - beta_tail: float
            - tail_fraction: float
        eps: Small value for numerical stability

    Returns:
        Tuple containing:
            - Total loss
            - Dictionary of partial losses
    """
    # Extract parameters from config
    correlation_targets = config["correlation_targets"]
    hill_targets = config["hill_exponent_targets"]
    io_target = config["io_matrix_target"]
    group_matrix = config["group_matrix"]
    weights = config["loss_weights"]
    beta_degree = config["beta_degree"]
    beta_tail = config["beta_tail"]
    tail_fraction = config["tail_fraction"]
    num_points = config.get("num_ccdf_points", 20)
    lambda_line = config.get("lambda_line", 0.1)

    # Compute partial losses
    correlation_loss, corr_losses = compute_correlation_loss(
        M, correlation_targets, beta_degree, eps
    )
    hill_loss, hill_losses = compute_hill_losses(
        M, hill_targets, beta_degree, beta_tail, tail_fraction, num_points, lambda_line, eps
    )
    io_loss = compute_io_loss(M, group_matrix, io_target, eps)
    smoothness_loss, smooth_losses = compute_smoothness_loss(
        M, beta_degree, beta_tail, num_points, eps
    )

    # Use more stable normalization
    with torch.no_grad():
        corr_scale = 1.0 + correlation_loss.detach()
        hill_scale = 1.0 + hill_loss.detach()
        io_scale = 1.0 + io_loss.detach()
        smooth_scale = 1.0 + smoothness_loss.detach()

    # Combine normalized losses with weights
    weighted_loss = (
        weights["correlation"] * correlation_loss / corr_scale
        + weights["hill"] * hill_loss / hill_scale
        + weights["io"] * io_loss / io_scale
        + weights["smooth"] * smoothness_loss / smooth_scale
    )
    weights_sum = sum(weights.values())
    total_loss = weighted_loss / weights_sum

    # Collect all partial losses
    partial_losses = {
        "correlation": correlation_loss,
        "hill": hill_loss,
        "io": io_loss,
        "smooth": smoothness_loss,
        **{f"correlation_{k}": v for k, v in corr_losses.items()},
        **{f"hill_{k}": v for k, v in hill_losses.items()},
        **{f"smooth_{k}": v for k, v in smooth_losses.items()},
    }

    # Log total loss
    logger.debug(
        f"Total loss = {total_loss.item():.3f} "
        f"(corr = {correlation_loss.item():.3f}, "
        f"hill = {hill_loss.item():.3f}, "
        f"io = {io_loss.item():.3f}, "
        f"smooth = {smoothness_loss.item():.3f})"
    )

    return total_loss, partial_losses
