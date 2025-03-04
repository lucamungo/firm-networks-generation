"""Video recording wrapper for network training.

This module provides functions to record the evolution of network distributions
during training without modifying the core training functions.
"""

import json
import tempfile
from pathlib import Path

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from network_generation.model import NetworkGenerator
from scripts.utils.viz import plot_distributions


def record_training_with_checkpoints(
    config_path, num_frames, save_dir, train_function, train_args, train_kwargs
):
    """Record training by saving checkpoints at regular intervals.

    Instead of modifying the training function, this approach:
    1. Creates a temporary checkpoint directory
    2. Runs short training segments and saves checkpoints
    3. Generates video from the checkpoints

    Args:
        config_path: Path to configuration file
        num_frames: Number of frames to capture
        save_dir: Directory to save checkpoints and video
        train_function: The training function to use
        train_args: Positional arguments for train_function
        train_kwargs: Keyword arguments for train_function

    Returns:
        Tuple containing:
            - Trained model
            - Training history
            - Path to video file
    """
    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Load configuration
    if isinstance(config_path, dict):
        # Config is already a dictionary
        config = config_path
    else:
        # Load config from file
        config_path = Path(config_path)
        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
            # Convert lists to tensors where needed
            if "group_matrix" in config and isinstance(config["group_matrix"], list):
                config["group_matrix"] = torch.tensor(config["group_matrix"])
            if "io_matrix_target" in config and isinstance(config["io_matrix_target"], list):
                config["io_matrix_target"] = torch.tensor(config["io_matrix_target"])
        else:
            # Assume it's a torch saved file
            config = torch.load(config_path)

    # Calculate total epochs
    total_epochs = 0
    if "training_phases" in config:
        for phase in config["training_phases"]:
            total_epochs += phase.get("epochs", 0)
        total_epochs *= train_kwargs.get("num_cycles", 1)
    else:
        total_epochs = config.get("num_epochs", 1000)

    # Calculate how many epochs to run between checkpoints
    epochs_per_segment = max(1, total_epochs // num_frames)

    # Initialize model from config
    model = NetworkGenerator(config)

    # Create the model for the initial state
    initial_model = NetworkGenerator(config)
    initial_weights = initial_model.get_network_weights()

    # Keep track of total epochs processed
    epochs_completed = 0
    all_history = {
        "total": [],
        "correlation": [],
        "hill": [],
        "io": [],
        "smooth": [],
        "density": [],
        "learning_rate": [],
    }

    # Set up progress tracking
    progress_bar = tqdm(total=total_epochs, desc="Training with checkpoints")

    # First checkpoint - initial state
    init_checkpoint_path = checkpoint_dir / f"checkpoint_{0:06d}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "loss": 0.0,
        },
        init_checkpoint_path,
    )

    # Create a temporary directory for config files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Main training loop with checkpoints
        while epochs_completed < total_epochs:
            # Calculate epochs for this segment
            epochs_this_segment = min(epochs_per_segment, total_epochs - epochs_completed)

            # Create a modified config for this segment
            segment_config = config.copy()

            if "training_phases" in config:
                # For progressive training, adjust epochs in each phase
                segment_config["training_phases"] = []
                remaining_epochs = epochs_this_segment

                for phase in config["training_phases"]:
                    if remaining_epochs <= 0:
                        break

                    phase_copy = phase.copy()
                    orig_epochs = phase.get("epochs", 0)

                    if remaining_epochs < orig_epochs:
                        phase_copy["epochs"] = remaining_epochs
                        segment_config["training_phases"].append(phase_copy)
                        remaining_epochs = 0
                    else:
                        segment_config["training_phases"].append(phase_copy)
                        remaining_epochs -= orig_epochs
            else:
                # For simple training, just set num_epochs
                segment_config["num_epochs"] = epochs_this_segment

            # Save this segment's config to a temporary file
            segment_config_json = segment_config.copy()
            # Convert tensors to lists for JSON serialization
            if "group_matrix" in segment_config_json and torch.is_tensor(
                segment_config_json["group_matrix"]
            ):
                segment_config_json["group_matrix"] = segment_config_json["group_matrix"].tolist()
            if "io_matrix_target" in segment_config_json and torch.is_tensor(
                segment_config_json["io_matrix_target"]
            ):
                segment_config_json["io_matrix_target"] = segment_config_json[
                    "io_matrix_target"
                ].tolist()

            segment_config_path = temp_dir / f"segment_config_{epochs_completed}.json"
            with open(segment_config_path, "w") as f:
                json.dump(segment_config_json, f)

            # Prepare arguments for the training function
            segment_train_args = [segment_config_path] + list(train_args[1:])

            # Load model state into a newly created model to avoid gradient history issues
            temp_model = NetworkGenerator(segment_config)
            temp_model.load_state_dict(model.state_dict())

            # Run training for this segment
            temp_model, segment_history = train_function(*segment_train_args, **train_kwargs)

            # Update model state
            model.load_state_dict(temp_model.state_dict())

            # Update total epochs processed
            epochs_completed += epochs_this_segment

            # Append history
            for key in segment_history:
                if key in all_history:
                    all_history[key].extend(segment_history[key])

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f"checkpoint_{epochs_completed:06d}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epochs_completed,
                    "loss": segment_history["total"][-1] if segment_history["total"] else 0.0,
                },
                checkpoint_path,
            )

            # Update progress
            progress_bar.update(epochs_this_segment)

    progress_bar.close()

    # Generate video from checkpoints
    video_path = generate_video_from_checkpoints(
        checkpoint_dir=checkpoint_dir,
        config=config,
        original_network=None,  # Will need to be provided separately
        initial_network=initial_weights,
        save_dir=save_dir,
    )

    return model, all_history, video_path


def generate_video_from_checkpoints(
    checkpoint_dir,
    config,
    original_network=None,
    initial_network=None,
    save_dir="./training_video",
    fps=10,
):
    """Generate a video from model checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get all checkpoint files
    checkpoint_files = sorted(list(checkpoint_dir.glob("checkpoint_*.pt")))
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    print(f"Found {len(checkpoint_files)} checkpoint files")

    # Prepare video writer
    video_path = save_dir / "training_evolution.mp4"
    writer = imageio.get_writer(video_path, fps=fps)

    # Extract parameters for plot_distributions
    beta_degree = config.get("beta_degree", 10.0)
    threshold_degree = config.get("threshold_degree", 1e-1)
    tail_fraction = config.get("tail_fraction", 0.05)

    # Create plots for each checkpoint and add to video
    plt.ioff()  # Turn off interactive mode
    progress_bar = tqdm(checkpoint_files, desc="Generating video frames")

    for checkpoint_file in progress_bar:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file)
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"] if "loss" in checkpoint else 0.0

        # Create model and load state
        model = NetworkGenerator(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Generate network weights
        generated_network = model.get_network_weights()

        # Create distribution plots
        fig = plot_distributions(
            W_original=original_network,
            W_generated=generated_network,
            W_initial=initial_network,
            beta_degree=beta_degree,
            threshold_degree=threshold_degree,
            tail_fraction=tail_fraction,
        )

        # Add epoch and loss information
        fig.suptitle(f"Epoch {epoch} - Loss: {loss:.4f}", fontsize=16)

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
