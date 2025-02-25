from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from network_generation.stats import compute_degrees, compute_strengths


def plot_distributions(
    W_original,
    W_generated,
    W_initial=None,
    beta_degree=10.0,
    threshold_degree: float = 1e-1,
    tail_fraction: float = 0.05,
):
    """Plot degree and strength distributions for original, initial, and generated networks."""
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Network Distributions Comparison", fontsize=14)

    # Function to create CCDF and find threshold point
    def compute_ccdf_numpy(values):
        sorted_values = np.sort(values)
        p = 1 - np.arange(len(sorted_values)) / len(sorted_values)

        # Calculate the index corresponding to the tail fraction
        k = max(int(len(values) * tail_fraction), 2)
        threshold_idx = len(values) - k
        threshold_value = sorted_values[threshold_idx]
        threshold_p = p[threshold_idx]

        return sorted_values, p, threshold_value, threshold_p

    # Compute distributions with Hill thresholds
    # Original network distributions
    in_degrees_orig = (W_original > 0).sum(dim=0).detach().numpy()
    out_degrees_orig = (W_original > 0).sum(dim=1).detach().numpy()
    in_strengths_orig = (W_original * (W_original > 0)).sum(dim=0).detach().numpy()
    out_strengths_orig = (W_original * (W_original > 0)).sum(dim=1).detach().numpy()

    # Generated network distributions
    M_generated = torch.log(W_generated)
    M_generated[M_generated < threshold_degree] = -20

    # in_degrees_gen = (
    #     compute_degrees(M=M_generated, beta_degree=beta_degree, threshold=threshold_degree, dim=0)
    #     .detach()
    #     .numpy()
    # )
    # out_degrees_gen = (
    #     compute_degrees(M=M_generated, beta_degree=beta_degree, threshold=threshold_degree, dim=1)
    #     .detach()
    #     .numpy()
    # )
    # in_strengths_gen = (
    #     compute_strengths(M_generated, beta=beta_degree, threshold=threshold_degree, dim=0)
    #     .detach()
    #     .numpy()
    # )
    # out_strengths_gen = (
    #     compute_strengths(M_generated, beta=beta_degree, threshold=threshold_degree, dim=1)
    #     .detach()
    #     .numpy()
    # )

    in_degrees_gen = (M_generated > threshold_degree).sum(dim=0).detach().numpy()
    out_degrees_gen = (M_generated > threshold_degree).sum(dim=1).detach().numpy()
    in_strengths_gen = ((M_generated > threshold_degree) * W_generated).sum(dim=0).detach().numpy()
    out_strengths_gen = ((M_generated > threshold_degree) * W_generated).sum(dim=1).detach().numpy()

    in_degrees_gen = in_degrees_gen[in_degrees_gen > 0]
    out_degrees_gen = out_degrees_gen[out_degrees_gen > 0]
    in_strengths_gen = in_strengths_gen[in_strengths_gen > 0]
    out_strengths_gen = out_strengths_gen[out_strengths_gen > 0]

    # Function to plot distribution with Hill threshold markers
    def plot_dist(ax, orig_values, gen_values, init_values=None, title=""):
        x_orig, y_orig, thresh_orig, p_orig = compute_ccdf_numpy(orig_values)
        x_gen, y_gen, thresh_gen, p_gen = compute_ccdf_numpy(gen_values)

        # Plot distributions
        line1 = ax.loglog(x_orig, y_orig, "b-", label="Original")[0]
        line2 = ax.loglog(x_gen, y_gen, "r--", label="Generated")[0]

        # Plot Hill threshold points with markers
        ax.plot(thresh_orig, p_orig, "x", color="gray", markersize=8)
        ax.plot(thresh_gen, p_gen, "x", color="gray", markersize=8)

        if init_values is not None:
            x_init, y_init, thresh_init, p_init = compute_ccdf_numpy(init_values)
            line3 = ax.loglog(x_init, y_init, "g:", label="Initial")[0]
            ax.plot(thresh_init, p_init, "x", color="gray", markersize=8)

        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        return line1, line2, line3 if init_values is not None else None

    # Plot all distributions
    if W_initial is not None:
        M_initial = torch.log(W_initial)
        in_degree_initial = (M_initial > threshold_degree).sum(dim=0).detach().numpy()
        out_degree_initial = (M_initial > threshold_degree).sum(dim=1).detach().numpy()
        in_strengths_initial = ((M_initial > threshold_degree) * W_initial).sum(dim=0).detach().numpy()
        out_strengths_initial = (
            ((M_initial > threshold_degree) * W_initial).sum(dim=1).detach().numpy()
        )
    else:
        in_degree_initial = None
        out_degree_initial = None
        in_strengths_initial = None
        out_strengths_initial = None

    lines = plot_dist(
        axes[0, 0],
        in_degrees_orig,
        in_degrees_gen,
        init_values=in_degree_initial,
        title="In-degree Distribution",
    )
    axes[0, 0].set_xlabel("In-degree")
    axes[0, 0].set_ylabel("CCDF")

    plot_dist(
        axes[0, 1],
        out_degrees_orig,
        out_degrees_gen,
        init_values=out_degree_initial,
        title="Out-degree Distribution",
    )
    axes[0, 1].set_xlabel("Out-degree")
    axes[0, 1].set_ylabel("CCDF")

    plot_dist(
        axes[1, 0],
        in_strengths_orig,
        in_strengths_gen,
        init_values=in_strengths_initial,
        title="In-strength Distribution",
    )
    axes[1, 0].set_xlabel("In-strength")
    axes[1, 0].set_ylabel("CCDF")

    plot_dist(
        axes[1, 1],
        out_strengths_orig,
        out_strengths_gen,
        init_values=out_strengths_initial,
        title="Out-strength Distribution",
    )
    axes[1, 1].set_xlabel("Out-strength")
    axes[1, 1].set_ylabel("CCDF")

    # Create a single legend at the bottom
    lines_for_legend = [line for line in lines if line is not None]
    # Add a dummy line for the threshold marker
    threshold_marker = plt.Line2D(
        [0], [0], color="gray", marker="x", linestyle="None", markersize=8, label="Hill threshold"
    )
    lines_for_legend.append(threshold_marker)

    fig.legend(handles=lines_for_legend, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=4)

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    return fig


def generate_plots(W_generated, W_original, W_initial, config, history):
    fig = plot_distributions(
        W_original,
        W_generated,
        W_initial,
        beta_degree=config["beta_degree"],
        threshold_degree=config["threshold_degree"],
        tail_fraction=config["tail_fraction"],
    )
    # Save plot
    plot_path = Path("./distributions.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nDistribution plots saved to {plot_path}")
    # Save training history plot
    plt.figure(figsize=(12, 6))
    plt.semilogy(history["total"], label="Total Loss")
    plt.semilogy(history["correlation"], label="Correlation Loss")
    plt.semilogy(history["hill"], label="Hill Loss")
    plt.semilogy(history["io"], label="IO Loss")
    plt.semilogy(history["smooth"], label="Smoothness Loss")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    # Save training history plot
    history_path = Path("./training_history.png")
    plt.savefig(history_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to {history_path}")
