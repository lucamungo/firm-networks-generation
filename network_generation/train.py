"""Training module for network generation.

This module implements the training loop for optimizing the network generator
model parameters to satisfy the desired constraints.
"""

import logging
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from .losses import compute_loss
from .model import NetworkGenerator
from .parse_config import parse_config

logger = logging.getLogger(__name__)


def train_model(
    config_path: str | Path,
    save_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[NetworkGenerator, dict[str, list[float]]]:
    """Train network generator model.

    Args:
        config_path: Path to config file
        save_path: Optional path to save final model
        device: Device to train on

    Returns:
        Trained model and dictionary of loss history
    """
    # Parse config
    config = parse_config(config_path)
    device = torch.device(device)

    # Create model and move to device
    model = NetworkGenerator(config).to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6
    )

    # Initialize loss history
    history = {
        "total": [],
        "correlation": [],
        "hill": [],
        "io": [],
        "smooth": [],
        "correlation_log_in_strength_out_strength": [],
        "correlation_log_in_degree_out_degree": [],
        "correlation_log_out_strength_out_degree": [],
        "hill_in_degree": [],
        "hill_out_degree": [],
        "hill_in_strength": [],
        "hill_out_strength": [],
        "smooth_in_degree": [],
        "smooth_out_degree": [],
        "smooth_in_strength": [],
        "smooth_out_strength": [],
        "learning_rate": [],
    }

    # Keep track of best model
    best_loss = float("inf")
    best_state_dict = None

    # Training loop
    logger.info("Starting training...")
    progress_bar = tqdm(range(config["num_epochs"]), desc="Training")

    for epoch in progress_bar:
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass and compute loss
        log_weights = model()
        loss, partial_losses = compute_loss(log_weights, config)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))

        # Update weights
        optimizer.step()

        # Record history
        current_lr = optimizer.param_groups[0]["lr"]
        history["total"].append(loss.item())
        history["learning_rate"].append(current_lr)
        for name, value in partial_losses.items():
            if name in history:
                history[name].append(value.item())

        # Update scheduler
        scheduler.step(loss)

        # Update best model if needed
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state_dict = model.state_dict().copy()

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
                "corr": f"{partial_losses['correlation'].item():.4f}",
                "hill": f"{partial_losses['hill'].item():.4f}",
                "io": f"{partial_losses['io'].item():.4f}",
                "smooth": f"{partial_losses['smooth'].item():.4f}",
            }
        )

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best model with loss {best_loss:.4f}")

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": (
                    best_state_dict if best_state_dict is not None else model.state_dict()
                ),
                "config": config,
                "final_loss": best_loss,
                "history": history,
            },
            save_path,
        )

    return model, history


def train_model_progressive(
    config_path: str | Path,
    save_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    num_cycles: int = 3,
) -> tuple[NetworkGenerator, dict[str, list[float]]]:
    """Train network generator model using progressive loss optimization."""
    config = parse_config(config_path)
    device = torch.device(device)

    # Create model and move to device
    model = NetworkGenerator(config).to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Get training phases from config
    if "training_phases" not in config:
        raise ValueError("Config must contain 'training_phases' section")

    phases = config["training_phases"]

    # Initialize history
    history = {
        "total": [],
        "correlation": [],
        "hill": [],
        "io": [],
        "smooth": [],
        "correlation_log_in_strength_out_strength": [],
        "correlation_log_in_degree_out_degree": [],
        "correlation_log_out_strength_out_degree": [],
        "hill_in_degree": [],
        "hill_out_degree": [],
        "hill_in_strength": [],
        "hill_out_strength": [],
        "smooth_in_degree": [],
        "smooth_out_degree": [],
        "smooth_in_strength": [],
        "smooth_out_strength": [],
        "learning_rate": [],
        "cycle": [],
        "phase": [],
    }

    best_loss = float("inf")
    best_state_dict = None

    # Calculate total epochs
    total_epochs = sum(phase["epochs"] for phase in phases) * num_cycles
    progress_bar = tqdm(total=total_epochs, desc="Training")
    current_epoch = 0

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.75,
        patience=50,
        min_lr=1e-6,
        threshold=1e-4,
        threshold_mode="rel",
    )

    # Training cycles
    for cycle in range(num_cycles):

        for phase_idx, phase in enumerate(phases):
            # Create phase config
            phase_config = config.copy()
            phase_weights = {k: 0.0 for k in config["loss_weights"].keys()}
            phase_weights.update(phase["weights"])
            phase_config["loss_weights"] = phase_weights

            # Training loop
            for epoch in range(phase["epochs"]):
                optimizer.zero_grad()
                log_weights = model()
                loss, partial_losses = compute_loss(log_weights, phase_config)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                optimizer.step()

                # Record history
                current_lr = optimizer.param_groups[0]["lr"]
                history["total"].append(loss.item())
                history["learning_rate"].append(current_lr)
                history["cycle"].append(cycle)
                history["phase"].append(phase_idx)
                for name, value in partial_losses.items():
                    if name in history:
                        history[name].append(value.item())

                scheduler.step(loss)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state_dict = model.state_dict().copy()

                # Update progress bar with consistent entries
                progress_bar.set_postfix(
                    {
                        "cycle": f"{cycle}/{num_cycles - 1}",
                        "phase": f"{phase_idx}/4",
                        "loss": f"{loss.item():.4f}",
                        "correlation": f"{partial_losses['correlation'].item():.4f}",
                        "hill": f"{partial_losses['hill'].item():.4f}",
                        "io": f"{partial_losses['io'].item():.4f}",
                        "smooth": f"{partial_losses['smooth'].item():.4f}",
                        "lr": f"{current_lr:.2e}",
                    }
                )
                progress_bar.update(1)
                current_epoch += 1

    progress_bar.close()

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best model with loss {best_loss:.4f}")

    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": (
                    best_state_dict if best_state_dict is not None else model.state_dict()
                ),
                "config": config,
                "final_loss": best_loss,
                "history": history,
            },
            save_path,
        )

    return model, history
