"""Training module for network generation.

This module implements the training loop for optimizing the network generator
model parameters to satisfy the desired constraints.
"""

import logging
from pathlib import Path
from typing import Optional

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
                "density": f"{partial_losses['density'].item():.4f}",
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


def update_history(
    history: dict[str, list[float]],
    loss: torch.Tensor,
    partial_losses: dict[str, torch.Tensor],
    current_lr: float,
    cycle: int,
    phase: int,
) -> None:
    """Update training history with current metrics."""
    history["total"].append(loss.item())
    history["learning_rate"].append(current_lr)
    history["cycle"].append(cycle)
    history["phase"].append(phase)

    for name, value in partial_losses.items():
        if name in history:
            history[name].append(value.item())


def save_model(
    model: NetworkGenerator,
    config: dict,
    best_loss: float,
    history: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """Save model checkpoint with associated data."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "final_loss": best_loss,
            "history": history,
        },
        save_path,
    )


class EarlyStopping:
    """Early stopping handler to prevent overfitting."""

    def __init__(self, patience: int = 50, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_model_progressive(
    config_path: str | Path,
    save_path: Optional[str | Path] = None,
    device: str | torch.device = "cpu",
    num_cycles: int = 3,
    early_stopping_patience: int = 100,  # New parameter
) -> tuple[NetworkGenerator, dict[str, list[float]]]:
    """Train network generator model using progressive loss optimization.

    Args:
        config_path: Path to configuration file
        save_path: Optional path to save checkpoints
        device: Device to train on
        num_cycles: Number of training cycles

    Returns:
        Tuple containing:
            - Trained model
            - Training history dictionary
    """
    # Parse config and setup
    config = parse_config(config_path)
    device = torch.device(device)

    # Create model and move to device
    model = NetworkGenerator(config).to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Get training phases
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

    # Calculate total epochs for progress bar
    total_epochs = sum(phase["epochs"] for phase in phases) * num_cycles
    progress_bar = tqdm(total=total_epochs, desc="Training")
    current_epoch = 0

    # Training cycles
    for cycle in range(num_cycles):
        for phase_idx, phase in enumerate(phases):
            # Create phase-specific config
            phase_config = config.copy()
            phase_weights = {k: 0.0 for k in config["loss_weights"].keys()}
            phase_weights.update(phase["weights"])
            phase_config["loss_weights"] = phase_weights

            # Create new optimizer and scheduler for this phase
            phase_lr = phase.get("learning_rate", config["learning_rate"])
            optimizer = optim.Adam(model.parameters(), lr=phase_lr)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.75,
                patience=max(20, phase["epochs"] // 10),  # Scale patience with phase length
                min_lr=1e-6,
                threshold=1e-4,
                threshold_mode="rel",
                verbose=True,
            )

            # Training loop for this phase
            for epoch in range(phase["epochs"]):
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass and compute loss
                log_weights = model()
                loss, partial_losses = compute_loss(log_weights, phase_config)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), phase_config.get("max_grad_norm", 1.0)
                )

                # Update weights
                optimizer.step()

                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]

                # Update history
                update_history(history, loss, partial_losses, current_lr, cycle, phase_idx)

                # Track best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state_dict = model.state_dict().copy()

                # Learning rate scheduling
                scheduler.step(loss)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "cycle": f"{cycle}/{num_cycles - 1}",
                        "phase": f"{phase_idx+1}/{len(phases)}",
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

            # Log phase completion
            logger.info(f"Completed phase {phase_idx} ({phase['name']})")

    progress_bar.close()

    # Restore overall best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best overall model with loss {best_loss:.4f}")

    # Save if requested
    if save_path is not None:
        save_model(model, config, best_loss, history, save_path)
        logger.info(f"Model saved to {save_path}")

    return model, history
