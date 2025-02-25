"""Example for network generation with updated methods.

This script demonstrates the use of the updated network generation model
with consistent log-weight density calculation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from macrocosm_visual.viz_setup import setup_matplotlib

from network_generation.hill_exponent import compute_hill_exponent
from network_generation.losses import compute_loss
from network_generation.model import CSHNetworkGenerator, NetworkGenerator
from network_generation.stats import (
    compute_degrees,
    compute_io_matrix,
    compute_log_correlation,
    compute_strengths,
)
from network_generation.train import train_model, train_model_progressive, train_model_progressive_with_dropout
from scripts.utils.analysis import print_comparison_table, print_loss_comparison
from scripts.utils.viz import generate_plots

setup_matplotlib()


def generate_synthetic_network(N=1000, density=0.1, mu=0.0, sigma=1.0):
    """Generate a synthetic network with log-normally distributed weights and specified density.

    Args:
        N (int): Number of nodes
        density (float): Target density level (0-1)
        mu (float): Mean of the underlying normal distribution
        sigma (float): Standard deviation of the underlying normal distribution

    Returns:
        torch.Tensor: Weight matrix with log-normally distributed weights
    """
    # Generate full matrix of log-normal weights
    normal_weights = torch.randn(N, N) * sigma + mu
    weights = torch.exp(normal_weights)

    # Normalize weights to have zero diagonal
    weights.fill_diagonal_(0)

    # Take row sums to compute total strength
    total_strength = weights.sum(dim=1)
    log_strength = torch.log(total_strength + 1e-8)

    # Sample log_degrees; log_degree and log_strength should have a correlation of 0.5
    log_degree = log_strength + 0.5 * torch.randn(N)
    degree = torch.exp(log_degree)

    # Rescale degrees; Max degree should be N / 10; min should be 1
    degree = degree / degree.max() * (N / 10)
    degree = torch.clamp(degree, 1, N)
    # Rescale so that the sum of the degrees is equal to int(N**2 * density)
    degree = degree / degree.sum() * (N**2 * density)
    degree = degree.int()

    mask = torch.zeros_like(weights)
    # For each column of mask, set degree[i] entries to 1 at random
    for i in range(N):
        mask[i, torch.randperm(N)[: degree[i]]] = 1

    # Apply mask
    weights = weights * mask

    return weights


def compute_network_properties(
    W, soft_approx: bool = True, beta_degree=10.0, threshold=1e-5, eps=1e-16
):
    """Compute various network properties."""
    # Compute degrees and strengths
    M = torch.log(W + eps)
    if soft_approx:
        in_degrees = compute_degrees(M, beta_degree, threshold=threshold, dim=0)
        out_degrees = compute_degrees(M, beta_degree, threshold=threshold, dim=1)
        in_strengths = compute_strengths(M, beta=beta_degree, threshold=threshold, dim=0)
        out_strengths = compute_strengths(M, beta=beta_degree, threshold=threshold, dim=1)
    else:
        in_degrees = (W > 0).sum(dim=0)
        out_degrees = (W > 0).sum(dim=1)
        in_strengths = W.sum(dim=0)
        out_strengths = W.sum(dim=1)

    # Compute correlations
    corr_in_out_str = compute_log_correlation(in_strengths, out_strengths)
    corr_in_out_deg = compute_log_correlation(in_degrees, out_degrees)
    corr_str_deg = compute_log_correlation(out_strengths, out_degrees)

    # Compute Hill exponents
    hill_in_degree = compute_hill_exponent(
        in_degrees, tail_fraction=0.05, beta_tail=10.0, temperature=0.05
    )
    hill_out_degree = compute_hill_exponent(
        out_degrees, tail_fraction=0.05, beta_tail=10.0, temperature=0.05
    )
    hill_in_strength = compute_hill_exponent(
        in_strengths, tail_fraction=0.05, beta_tail=10.0, temperature=0.05
    )
    hill_out_strength = compute_hill_exponent(
        out_strengths, tail_fraction=0.05, beta_tail=10.0, temperature=0.05
    )

    # Create simple industry grouping (2 groups of equal size)
    N = W.shape[0]
    group_assignments = {"0": list(range(N // 2)), "1": list(range(N // 2, N))}

    # Create group matrix
    group_matrix = torch.zeros((2, N))
    for i, firms in group_assignments.items():
        group_matrix[int(i), firms] = 1

    # Compute IO matrix
    io_matrix = compute_io_matrix(torch.log(W), group_matrix)

    # Compute density
    density = in_degrees.sum() / (N * N) * 100

    return {
        "correlations": {
            "log_in_strength_out_strength": float(corr_in_out_str),
            "log_in_degree_out_degree": float(corr_in_out_deg),
            "log_out_strength_out_degree": float(corr_str_deg),
        },
        "hill_exponents": {
            "in_degree": float(hill_in_degree),
            "out_degree": float(hill_out_degree),
            "in_strength": float(hill_in_strength),
            "out_strength": float(hill_out_strength),
        },
        "io_matrix": io_matrix.tolist(),
        "density": float(density),
        "group_assignments": group_assignments,
    }


def create_config(properties, N, M=10, num_epochs=3000, num_cycles=1):
    """Create configuration dictionary.

    Args:
        properties: Network properties dictionary
        N: Number of nodes
        M: Number of components for factorization (default: 10)
        num_epochs: Total number of epochs (default: 3000)
        num_cycles: Number of training cycles (default: 1)
    """
    # Create group matrix from assignments
    group_assignments = properties["group_assignments"]
    num_groups = len(group_assignments)
    group_matrix = torch.zeros((num_groups, N))
    for i, firms in group_assignments.items():
        group_matrix[int(i), firms] = 1

    # Calculate epochs per phase
    num_phases = 6  # Density, IO Matrix, Correlations, Hill Exponents, Smoothness, Fine Tuning
    epochs_per_phase = num_epochs // (num_phases * num_cycles)

    # Create config with tensor values converted to lists where needed
    config = {
        "N": N,
        "M": M,
        "group_assignments": properties["group_assignments"],
        "group_matrix": group_matrix,  # Keep as tensor for compute_loss
        "correlation_targets": properties["correlations"],
        "hill_exponent_targets": properties["hill_exponents"],
        "io_matrix_target": torch.tensor(properties["io_matrix"]),  # Convert back to tensor
        "density_target": properties["density"],
        "loss_weights": {
            "correlation": 1.0,
            "hill": 1.0,
            "io": 5.0,
            "smooth": 1.0,
            "density": 0.1,
            "continuity": 100.0,
            "disconnection": 1.0,
        },
        "training_phases": [
            {
                "name": "Density",
                "epochs": epochs_per_phase,
                "weights": {
                    "correlation": 0.0,
                    "hill": 0.0,
                    "io": 0.00,
                    "smooth": 1.0,
                    "density": 1.0,
                    "continuity": 0.0,
                    "disconnection": 1.0,
                },
            },
            {
                "name": "IO Matrix",
                "epochs": epochs_per_phase,
                "weights": {
                    "correlation": 0.0,
                    "hill": 0.0,
                    "io": 1.0,
                    "smooth": 1.0,
                    "density": 0.1,
                    "continuity": 0.0,
                    "disconnection": 1.0,
                },
            },
            {
                "name": "Correlations",
                "epochs": epochs_per_phase,
                "weights": {
                    "correlation": 1.0,
                    "hill": 0.0,
                    "io": 0.5,
                    "smooth": 1.0,
                    "density": 0.1,
                    "continuity": 0.0,
                    "disconnection": 1.0,
                },
            },
            {
                "name": "Hill Exponents",
                "epochs": epochs_per_phase,
                "weights": {
                    "correlation": 0.1,
                    "hill": 1.0,
                    "io": 0.5,
                    "smooth": 1.0,
                    "density": 0.1,
                    "continuity": 0.0,
                    "disconnection": 1.0,
                },
            },
            {
                "name": "Smoothness",
                "epochs": epochs_per_phase,
                "weights": {
                    "correlation": 0.1,
                    "hill": 1.0,
                    "io": 0.1,
                    "smooth": 10.0,
                    "density": 0.1,
                    "continuity": 0.0,
                    "disconnection": 1.0,
                },
            },
            {
                "name": "Fine Tuning",
                "epochs": epochs_per_phase,
                "weights": {
                    "correlation": 1.0,
                    "hill": 1.0,
                    "io": 1.0,
                    "smooth": 1.0,
                    "density": 0.1,
                    "continuity": 1.0,
                    "disconnection": 1.0,
                },
            },
        ],
        "learning_rate": 0.01,
        "num_epochs": num_epochs,
        "beta_degree": 5.0,
        "threshold_degree": 0.0,
        "beta_ccdf": 5.0,
        "beta_tail": 10.0,
        "tail_fraction": 0.05,
        "num_ccdf_points": 30,
    }

    # Create a JSON-serializable version of the config
    json_config = config.copy()
    json_config["group_matrix"] = group_matrix.tolist()
    json_config["io_matrix_target"] = properties["io_matrix"]  # Already a list

    return config, json_config


def main():
    # Add argument parsing
    parser = argparse.ArgumentParser(description="Network generation training script.")
    parser.add_argument("--N", type=int, default=1000, help="Number of nodes (default: 1000)")
    parser.add_argument("--M", type=int, default=10, help="Number of components (default: 10)")
    parser.add_argument(
        "--sparsity", type=float, default=0.05, help="Network sparsity (default: 0.05)"
    )
    parser.add_argument(
        "--progressive", action="store_true", help="Use progressive training with multiple cycles"
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=1,
        help="Number of cycles for progressive training (default: 1)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Total number of training epochs (default: 1000)"
    )
    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Parameters
    N = args.N  # Number of nodes
    sparsity = args.sparsity  # Network sparsity
    M = args.M  # Number of components for factorization

    print("Step 1: Generating synthetic network...")
    W_original = generate_synthetic_network(N, density=sparsity)

    print("Step 2: Computing network properties...")
    properties = compute_network_properties(
        W_original, soft_approx=True, beta_degree=5.0, threshold=0.0
    )

    print("\nStep 3: Creating configuration...")
    if args.progressive:
        print(f"Using progressive training with {args.num_cycles} cycles...")
        print(
            f"Total epochs: {args.epochs} (≈ {args.epochs // (6 * args.num_cycles)} epochs per phase)"
        )
        config, json_config = create_config(
            properties, N, num_epochs=args.epochs, num_cycles=args.num_cycles, M=M
        )
    else:
        print(f"Using standard training with {args.epochs} epochs")
        config, json_config = create_config(properties, N, num_epochs=args.epochs, M=M)

    # Save configuration (use JSON-serializable version)
    config_path = Path("synthetic_config.json")
    with open(config_path, "w") as f:
        json.dump(json_config, f, indent=2)

    # Initialize model
    initial_model = NetworkGenerator(config)
    initial_log_weights = initial_model()
    W_initial = initial_model.get_network_weights()

    # Compute initial properties
    initial_properties = compute_network_properties(
        W_initial,
        soft_approx=True,
        beta_degree=config["beta_degree"],
        threshold=config["threshold_degree"],
    )

    # Compute initial loss
    initial_loss, initial_partial_losses = compute_loss(initial_log_weights, config)

    print("\nInitial Density:")
    print(f"Target: {properties['density']:.4f}%, Initial: {initial_properties['density']:.4f}%")

    print("\nStep 4: Training model...")
    if args.progressive:
        model, history = train_model_progressive_with_dropout(config_path, num_cycles=args.num_cycles)
    else:
        model, history = train_model(config_path)

    print("\nStep 5: Evaluating results...")
    final_log_weights = model()
    W_generated = model.get_network_weights()

    final_properties = compute_network_properties(
        W_generated,
        soft_approx=True,
        beta_degree=config["beta_degree"],
        threshold=config["threshold_degree"],
    )

    # Compute final losses
    final_loss, final_partial_losses = compute_loss(final_log_weights, config)

    # Print formatted results
    print("\n=== RESULTS SUMMARY ===")

    print_comparison_table(
        "Network Correlations",
        properties["correlations"],
        final_properties["correlations"],
        difference=True,
    )

    print_comparison_table(
        "Hill Exponents",
        properties["hill_exponents"],
        final_properties["hill_exponents"],
        difference=True,
    )

    # Add density comparison
    print("\nNetwork Density:")
    print("-" * 70)
    print(f"{'Property':<30} {'Original':>12} {'Generated':>12} {'Diff':>12}")
    print("-" * 70)
    orig_density = properties["density"]
    gen_density = final_properties["density"]
    diff_density = gen_density - orig_density
    print(f"{'density':<30} {orig_density:>12.3f} {gen_density:>12.3f} {diff_density:>12.3f}")
    print("-" * 70)

    # Print loss comparison
    print_loss_comparison(initial_partial_losses, final_partial_losses)

    # Print IO matrix comparison
    print("\nIO Matrix Comparison:")
    print("\nOriginal IO Matrix:")
    io_original = np.array(properties["io_matrix"])
    print(np.array2string(io_original, precision=3, suppress_small=True))

    print("\nGenerated IO Matrix:")
    io_generated = np.array(final_properties["io_matrix"])
    print(np.array2string(io_generated, precision=3, suppress_small=True))

    print("\nIO Matrix Difference (Generated - Original):")
    io_diff = io_generated - io_original
    print(np.array2string(io_diff, precision=3, suppress_small=True))

    # Plot distributions
    generate_plots(W_generated, W_original, W_initial, config, history)


if __name__ == "__main__":
    main()
