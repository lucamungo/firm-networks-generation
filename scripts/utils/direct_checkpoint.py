"""Direct checkpoint extension for train_model_progressive_with_dropout.

This file adds checkpoint recording capabilities directly to your training function
without modifying its core behavior.
"""

import shutil
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from network_generation.model import NetworkGenerator
from network_generation.parse_config import parse_config
from network_generation.train import train_model_progressive_with_dropout


def train_with_checkpoints(
    config_path,
    save_dir="./training_video",
    num_frames=50,
    fps=8,  # Lower FPS means frames display longer (25% longer => reduce FPS by 20%)
    original_network=None,
    initial_network=None,
    cleanup_checkpoints=True,  # New parameter to control checkpoint cleanup
    **training_kwargs,
):
    """Train and record checkpoints for video generation.

    This function is a thin wrapper around train_model_progressive_with_dropout
    that adds checkpoint recording.

    Args:
        config_path: Path to configuration file
        save_dir: Directory to save checkpoint files and video
        num_frames: Number of frames to record for the video
        fps: Frames per second for the output video
        original_network: Original network for comparison
        initial_network: Initial network for comparison
        cleanup_checkpoints: Whether to delete checkpoint files after video creation
        **training_kwargs: Additional arguments for train_model_progressive_with_dropout

    Returns:
        Tuple containing:
            - Trained model
            - Training history
            - Path to generated video
    """
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Load configuration
    config = parse_config(config_path)

    # Calculate total epochs based on ACTIVE phases
    total_epochs = 0
    if "training_phases" in config:
        # Count only epochs from active phases (those not commented out)
        for phase in config["training_phases"]:
            total_epochs += phase.get("epochs", 0)
        total_epochs *= training_kwargs.get("num_cycles", 1)
    else:
        total_epochs = config.get("num_epochs", 1000)

    print(f"Detected total epochs: {total_epochs}")

    # Calculate checkpoint intervals
    checkpoint_interval = max(1, total_epochs // num_frames)
    print(f"Will save a checkpoint every {checkpoint_interval} epochs")

    # Save initial model state (epoch 0)
    initial_model = NetworkGenerator(config)
    init_checkpoint_path = checkpoint_dir / f"checkpoint_{0:06d}.pt"
    torch.save(
        {
            "model_state_dict": initial_model.state_dict(),
            "epoch": 0,
            "loss": 0.0,
            "cycle": 0,
            "phase": 0,
            "phase_name": "Initial",
        },
        init_checkpoint_path,
    )

    # Extract known parameters from training_kwargs
    device = training_kwargs.pop("device", "cpu")
    num_cycles = training_kwargs.pop("num_cycles", 3)
    early_stopping_patience = training_kwargs.pop("early_stopping_patience", 100)
    dropout_rate = training_kwargs.pop("dropout_rate", 0.2)
    component_dropout_rate = training_kwargs.pop("component_dropout_rate", 0.1)
    use_block_update = training_kwargs.pop("use_block_update", False)
    block_update_cycle = training_kwargs.pop("block_update_cycle", 3)
    min_lr = training_kwargs.pop("min_lr", 5e-7)

    # Counter for checkpoint saving
    class CheckpointState:
        def __init__(self):
            self.count = 0
            self.saved_epochs = []
            self.current_cycle = 0
            self.current_phase = 0
            self.current_phase_name = "Initial"

    checkpoint_state = CheckpointState()

    # Monkey patch the update_history function to save checkpoints
    import network_generation.train as train_module

    original_update_history = train_module.update_history

    def patched_update_history(history, loss, partial_losses, current_lr, cycle, phase):
        # Call original function
        original_update_history(history, loss, partial_losses, current_lr, cycle, phase)

        # Update state
        checkpoint_state.current_cycle = cycle
        checkpoint_state.current_phase = phase

        # Find the phase name if possible
        if hasattr(config, "get") and callable(config.get) and "training_phases" in config:
            phases = config["training_phases"]
            if phase < len(phases) and "name" in phases[phase]:
                checkpoint_state.current_phase_name = phases[phase]["name"]
            else:
                checkpoint_state.current_phase_name = f"Phase {phase}"

        # Update counter and check if it's time to save a checkpoint
        checkpoint_state.count += 1
        if (
            checkpoint_state.count % checkpoint_interval == 0
            or checkpoint_state.count == total_epochs
        ):
            # Get model from the global frame
            for frame_info in inspect.stack():
                frame = frame_info.frame
                if "model" in frame.f_locals and isinstance(
                    frame.f_locals["model"], NetworkGenerator
                ):
                    model = frame.f_locals["model"]

                    # Save checkpoint
                    checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_state.count:06d}.pt"
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "epoch": checkpoint_state.count,
                            "loss": loss.item(),
                            "cycle": cycle,
                            "phase": phase,
                            "phase_name": checkpoint_state.current_phase_name,
                        },
                        checkpoint_path,
                    )

                    checkpoint_state.saved_epochs.append(checkpoint_state.count)
                    break

    # We need the inspect module for frame introspection
    import inspect

    # Apply the monkey patch
    train_module.update_history = patched_update_history

    try:
        # Run the training with the patched function
        model, history = train_model_progressive_with_dropout(
            config_path=config_path,
            save_path=None,  # We'll save our own checkpoints
            device=device,
            num_cycles=num_cycles,
            early_stopping_patience=early_stopping_patience,
            dropout_rate=dropout_rate,
            component_dropout_rate=component_dropout_rate,
            use_block_update=use_block_update,
            block_update_cycle=block_update_cycle,
            min_lr=min_lr,
            **training_kwargs,
        )
    finally:
        # Restore the original function
        train_module.update_history = original_update_history

    print(
        f"Training completed. Saved {len(checkpoint_state.saved_epochs)} checkpoints."
    )

    # Generate video from checkpoints
    video_path = generate_video_from_checkpoints(
        checkpoint_dir=checkpoint_dir,
        config=config,
        original_network=original_network,
        initial_network=initial_network,
        save_dir=save_dir,
        fps=fps,
    )

    # Cleanup checkpoints if requested
    if cleanup_checkpoints:
        print("Cleaning up checkpoint files...")
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"Checkpoint directory removed: {checkpoint_dir}")
        except Exception as e:
            print(f"Warning: Could not remove checkpoint directory: {e}")

    return model, history, video_path


def generate_video_from_checkpoints(
    checkpoint_dir,
    config,
    original_network=None,
    initial_network=None,
    save_dir="./training_video",
    fps=8,
):
    """Generate a video from model checkpoints with fixed axes."""
    checkpoint_dir = Path(checkpoint_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all checkpoint files
    checkpoint_files = sorted(list(checkpoint_dir.glob("checkpoint_*.pt")))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Extract parameters for plot_distributions
    beta_degree = config.get("beta_degree", 10.0)
    threshold_degree = config.get("threshold_degree", 1e-1)
    tail_fraction = config.get("tail_fraction", 0.05)

    # First pass: determine fixed axes limits by analyzing all data
    print("Analyzing checkpoints to determine fixed axes limits...")

    # Initialize min/max values for all four plots
    limits = {
        "in_degree": {
            "xmin": float("inf"),
            "xmax": float("-inf"),
            "ymin": float("inf"),
            "ymax": float("-inf"),
        },
        "out_degree": {
            "xmin": float("inf"),
            "xmax": float("-inf"),
            "ymin": float("inf"),
            "ymax": float("-inf"),
        },
        "in_strength": {
            "xmin": float("inf"),
            "xmax": float("-inf"),
            "ymin": float("inf"),
            "ymax": float("-inf"),
        },
        "out_strength": {
            "xmin": float("inf"),
            "xmax": float("-inf"),
            "ymin": float("inf"),
            "ymax": float("-inf"),
        },
    }

    # Function to compute CCDF and update limits
    def compute_ccdf_numpy(values, key):
        if len(values) == 0:
            return None, None, None, None

        sorted_values = np.sort(values)
        p = 1 - np.arange(len(sorted_values)) / len(sorted_values)

        # Update limits
        limits[key]["xmin"] = min(limits[key]["xmin"], sorted_values.min())
        limits[key]["xmax"] = max(limits[key]["xmax"], sorted_values.max())
        if p.min() > 0:
            limits[key]["ymin"] = min(limits[key]["ymin"], p.min())
        limits[key]["ymax"] = max(limits[key]["ymax"], p.max())

        # Calculate the index corresponding to the tail fraction
        k = max(int(len(values) * tail_fraction), 2)
        threshold_idx = len(values) - k
        if threshold_idx < len(sorted_values):
            threshold_value = sorted_values[threshold_idx]
            threshold_p = p[threshold_idx]
            return sorted_values, p, threshold_value, threshold_p
        return sorted_values, p, None, None

    # Analyze original network if provided
    if original_network is not None:
        M_original = torch.log(original_network + 1e-16)

        in_degrees_orig = (M_original > threshold_degree).sum(dim=0).detach().numpy()
        out_degrees_orig = (M_original > threshold_degree).sum(dim=1).detach().numpy()
        in_strengths_orig = (
            ((M_original > threshold_degree) * original_network).sum(dim=0).detach().numpy()
        )
        out_strengths_orig = (
            ((M_original > threshold_degree) * original_network).sum(dim=1).detach().numpy()
        )

        compute_ccdf_numpy(in_degrees_orig, "in_degree")
        compute_ccdf_numpy(out_degrees_orig, "out_degree")
        compute_ccdf_numpy(in_strengths_orig, "in_strength")
        compute_ccdf_numpy(out_strengths_orig, "out_strength")

    # Analyze initial network if provided
    if initial_network is not None:
        M_initial = torch.log(initial_network + 1e-16)

        in_degrees_init = (M_initial > threshold_degree).sum(dim=0).detach().numpy()
        out_degrees_init = (M_initial > threshold_degree).sum(dim=1).detach().numpy()
        in_strengths_init = (
            ((M_initial > threshold_degree) * initial_network).sum(dim=0).detach().numpy()
        )
        out_strengths_init = (
            ((M_initial > threshold_degree) * initial_network).sum(dim=1).detach().numpy()
        )

        compute_ccdf_numpy(in_degrees_init, "in_degree")
        compute_ccdf_numpy(out_degrees_init, "out_degree")
        compute_ccdf_numpy(in_strengths_init, "in_strength")
        compute_ccdf_numpy(out_strengths_init, "out_strength")

    # Analyze all checkpoints for min/max values
    for checkpoint_file in tqdm(checkpoint_files, desc="Analyzing checkpoints"):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)

        # Create model and load state
        model = NetworkGenerator(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate network weights
        generated_network = model.get_network_weights()
        M_generated = torch.log(generated_network + 1e-16)

        # Calculate distributions
        in_degrees_gen = (M_generated > threshold_degree).sum(dim=0).detach().numpy()
        out_degrees_gen = (M_generated > threshold_degree).sum(dim=1).detach().numpy()
        in_strengths_gen = (
            ((M_generated > threshold_degree) * generated_network).sum(dim=0).detach().numpy()
        )
        out_strengths_gen = (
            ((M_generated > threshold_degree) * generated_network).sum(dim=1).detach().numpy()
        )

        # Filter zero values
        in_degrees_gen = in_degrees_gen[in_degrees_gen > 0]
        out_degrees_gen = out_degrees_gen[out_degrees_gen > 0]
        in_strengths_gen = in_strengths_gen[in_strengths_gen > 0]
        out_strengths_gen = out_strengths_gen[out_strengths_gen > 0]

        # Update limits
        compute_ccdf_numpy(in_degrees_gen, "in_degree")
        compute_ccdf_numpy(out_degrees_gen, "out_degree")
        compute_ccdf_numpy(in_strengths_gen, "in_strength")
        compute_ccdf_numpy(out_strengths_gen, "out_strength")

    # Add some padding to the limits (10% on each side)
    for key in limits:
        x_range = limits[key]["xmax"] - limits[key]["xmin"]
        limits[key]["xmin"] = max(
            0.1, limits[key]["xmin"] - 0.05 * x_range
        )  # Don't go below 0.1 for log scale
        limits[key]["xmax"] = limits[key]["xmax"] + 0.05 * x_range

        # y-axis is typically 0 to 1 for CCDF, but add a little padding
        limits[key]["ymin"] = max(
            0.001, limits[key]["ymin"] * 0.9
        )  # Don't go below 0.001 for log scale
        limits[key]["ymax"] = min(1.0, limits[key]["ymax"] * 1.1)  # Don't exceed 1.0 for CCDF

    # Prepare video writer
    video_path = save_dir / "training_evolution.mp4"
    writer = imageio.get_writer(video_path, fps=fps)

    # Custom plot_distributions function with fixed axes
    def custom_plot_distributions(W_original, W_generated, W_initial, fixed_limits):
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Function to create CCDF with fixed axes
        def plot_dist(ax, orig_values, gen_values, init_values, title, key):
            if len(orig_values) == 0 or len(gen_values) == 0:
                return None, None

            # Original network
            x_orig = np.sort(orig_values)
            p_orig = 1 - np.arange(len(x_orig)) / len(x_orig)

            # Calculate Hill threshold for original network
            k_orig = max(int(len(orig_values) * tail_fraction), 2)
            thresh_idx_orig = len(orig_values) - k_orig
            if thresh_idx_orig < len(x_orig):
                thresh_orig = x_orig[thresh_idx_orig]
                p_thresh_orig = p_orig[thresh_idx_orig]

            # Generated network
            x_gen = np.sort(gen_values)
            p_gen = 1 - np.arange(len(x_gen)) / len(x_gen)

            # Calculate Hill threshold for generated network
            k_gen = max(int(len(gen_values) * tail_fraction), 2)
            thresh_idx_gen = len(gen_values) - k_gen
            if thresh_idx_gen < len(x_gen):
                thresh_gen = x_gen[thresh_idx_gen]
                p_thresh_gen = p_gen[thresh_idx_gen]

            # Plot with fixed axes
            line1 = ax.loglog(x_orig, p_orig, "b-", label="Original")[0]
            line2 = ax.loglog(x_gen, p_gen, "r--", label="Generated")[0]

            # Plot Hill thresholds with markers
            if thresh_idx_orig < len(x_orig):
                ax.plot(thresh_orig, p_thresh_orig, "x", color="gray", markersize=8)
            if thresh_idx_gen < len(x_gen):
                ax.plot(thresh_gen, p_thresh_gen, "x", color="gray", markersize=8)

            lines = [line1, line2]

            # Plot initial if provided
            if init_values is not None and len(init_values) > 0:
                x_init = np.sort(init_values)
                p_init = 1 - np.arange(len(x_init)) / len(x_init)

                # Calculate Hill threshold for initial network
                k_init = max(int(len(init_values) * tail_fraction), 2)
                thresh_idx_init = len(init_values) - k_init
                if thresh_idx_init < len(x_init):
                    thresh_init = x_init[thresh_idx_init]
                    p_thresh_init = p_init[thresh_idx_init]
                    ax.plot(thresh_init, p_thresh_init, "x", color="gray", markersize=8)

                line3 = ax.loglog(x_init, p_init, "g:", label="Initial")[0]
                lines.append(line3)

            # Set fixed axes
            ax.set_xlim(fixed_limits[key]["xmin"], fixed_limits[key]["xmax"])
            ax.set_ylim(fixed_limits[key]["ymin"], fixed_limits[key]["ymax"])

            ax.set_title(title)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            return lines

        # Process the data
        M_original = torch.log(W_original + 1e-16) if W_original is not None else None
        M_generated = torch.log(W_generated + 1e-16)
        M_initial = torch.log(W_initial + 1e-16) if W_initial is not None else None

        # Calculate distributions for each network
        if W_original is not None:
            in_degrees_orig = (M_original > threshold_degree).sum(dim=0).detach().numpy()
            out_degrees_orig = (M_original > threshold_degree).sum(dim=1).detach().numpy()
            in_strengths_orig = (
                ((M_original > threshold_degree) * W_original).sum(dim=0).detach().numpy()
            )
            out_strengths_orig = (
                ((M_original > threshold_degree) * W_original).sum(dim=1).detach().numpy()
            )
        else:
            in_degrees_orig = np.array([])
            out_degrees_orig = np.array([])
            in_strengths_orig = np.array([])
            out_strengths_orig = np.array([])

        # Generated network
        in_degrees_gen = (M_generated > threshold_degree).sum(dim=0).detach().numpy()
        out_degrees_gen = (M_generated > threshold_degree).sum(dim=1).detach().numpy()
        in_strengths_gen = (
            ((M_generated > threshold_degree) * W_generated).sum(dim=0).detach().numpy()
        )
        out_strengths_gen = (
            ((M_generated > threshold_degree) * W_generated).sum(dim=1).detach().numpy()
        )

        # Filter zero values
        in_degrees_gen = in_degrees_gen[in_degrees_gen > 0]
        out_degrees_gen = out_degrees_gen[out_degrees_gen > 0]
        in_strengths_gen = in_strengths_gen[in_strengths_gen > 0]
        out_strengths_gen = out_strengths_gen[out_strengths_gen > 0]

        # Initial network if provided
        if W_initial is not None:
            in_degrees_init = (M_initial > threshold_degree).sum(dim=0).detach().numpy()
            out_degrees_init = (M_initial > threshold_degree).sum(dim=1).detach().numpy()
            in_strengths_init = (
                ((M_initial > threshold_degree) * W_initial).sum(dim=0).detach().numpy()
            )
            out_strengths_init = (
                ((M_initial > threshold_degree) * W_initial).sum(dim=1).detach().numpy()
            )

            # Filter zero values
            in_degrees_init = in_degrees_init[in_degrees_init > 0]
            out_degrees_init = out_degrees_init[out_degrees_init > 0]
            in_strengths_init = in_strengths_init[in_strengths_init > 0]
            out_strengths_init = out_strengths_init[out_strengths_init > 0]
        else:
            in_degrees_init = None
            out_degrees_init = None
            in_strengths_init = None
            out_strengths_init = None

        # Plot distributions with fixed axes
        lines1 = plot_dist(
            axes[0, 0],
            in_degrees_orig,
            in_degrees_gen,
            in_degrees_init,
            "In-degree Distribution",
            "in_degree",
        )
        axes[0, 0].set_xlabel("In-degree")
        axes[0, 0].set_ylabel("CCDF")

        plot_dist(
            axes[0, 1],
            out_degrees_orig,
            out_degrees_gen,
            out_degrees_init,
            "Out-degree Distribution",
            "out_degree",
        )
        axes[0, 1].set_xlabel("Out-degree")
        axes[0, 1].set_ylabel("CCDF")

        plot_dist(
            axes[1, 0],
            in_strengths_orig,
            in_strengths_gen,
            in_strengths_init,
            "In-strength Distribution",
            "in_strength",
        )
        axes[1, 0].set_xlabel("In-strength")
        axes[1, 0].set_ylabel("CCDF")

        plot_dist(
            axes[1, 1],
            out_strengths_orig,
            out_strengths_gen,
            out_strengths_init,
            "Out-strength Distribution",
            "out_strength",
        )
        axes[1, 1].set_xlabel("Out-strength")
        axes[1, 1].set_ylabel("CCDF")

        # Create a single legend at the bottom
        if lines1:
            # Add a dummy line for the threshold marker
            threshold_marker = plt.Line2D(
                [0],
                [0],
                color="gray",
                marker="x",
                linestyle="None",
                markersize=8,
                label="Hill threshold",
            )
            valid_lines = [line for line in lines1 if line is not None]
            valid_lines.append(threshold_marker)

            fig.legend(handles=valid_lines, loc="center", bbox_to_anchor=(0.5, 0.02), ncol=4)

            # Adjust layout to make room for the legend
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)

        return fig

    # Create plots for each checkpoint and add to video
    plt.ioff()  # Turn off interactive mode
    progress_bar = tqdm(checkpoint_files, desc="Generating video frames")

    for checkpoint_file in progress_bar:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"] if "loss" in checkpoint else 0.0
        cycle = checkpoint.get("cycle", 0)
        phase = checkpoint.get("phase", 0)
        phase_name = checkpoint.get("phase_name", f"{phase}")

        # Create model and load state
        model = NetworkGenerator(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate network weights
        generated_network = model.get_network_weights()

        # Create distribution plots with fixed axes
        fig = custom_plot_distributions(
            W_original=original_network,
            W_generated=generated_network,
            W_initial=initial_network,
            fixed_limits=limits,
        )

        # Add epoch, cycle, phase and loss information to title, put a padding between the suptitle and the plots
        fig.subplots_adjust(top=0.9)
        if epoch == 0:
            fig.suptitle("Initial State", fontsize=16)
        else:
            title_text = (
                f"Epoch {epoch:<3d} - Cycle {cycle:<3d} - {phase_name:<10s} - Loss: {loss:.4f}"
            )
            fig.suptitle(title_text, fontsize=16)

        # Convert figure to image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())

        # Add to video
        writer.append_data(image)

        # Close figure to free memory
        plt.close(fig)

    # Close video writer
    writer.close()
    print(f"Video saved to {video_path}")

    return video_path
