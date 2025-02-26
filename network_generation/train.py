"""Training module for network generation.

This module implements the training loop for optimizing the network generator
model parameters to satisfy the desired constraints.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
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
    early_stopping_patience: int = 100,
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
        "density": [],  # Add density to history tracking
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

            # Check if density weight is zero in this phase
            density_weight = phase_weights.get("density", 0.0)
            density_training = density_weight > 0

            # Freeze/unfreeze model parameters based on density weight
            if hasattr(model, "freeze_density") and hasattr(model, "unfreeze_density"):
                # If model has explicit density control methods, use them
                if density_training:
                    model.unfreeze_density()
                    logger.info(
                        f"Phase {phase_idx} ({phase['name']}): Unfreezing density parameters"
                    )
                else:
                    model.freeze_density()
                    logger.info(f"Phase {phase_idx} ({phase['name']}): Freezing density parameters")
            else:
                # For standard NetworkGenerator, we can't specifically freeze density parameters
                # So we'll log a message but continue with all parameters
                if not density_training:
                    logger.info(
                        f"Phase {phase_idx} ({phase['name']}): Density weight is zero but model doesn't support parameter freezing"
                    )

            # Create new optimizer with only trainable parameters
            phase_lr = phase.get("learning_rate", config["learning_rate"])
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=phase_lr)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.75,
                patience=max(20, phase["epochs"] // 10),  # Scale patience with phase length
                min_lr=1e-6,
                threshold=1e-4,
                threshold_mode="rel",
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
                    [p for p in model.parameters() if p.requires_grad],
                    phase_config.get("max_grad_norm", 1.0),
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

                # Update progress bar with more detailed density info
                progress_bar.set_postfix(
                    {
                        "cycle": f"{cycle}/{num_cycles - 1}",
                        "phase": f"{phase_idx+1}/{len(phases)}",
                        "loss": f"{loss.item():.3f}",
                        "corr": f"{partial_losses['correlation'].item():.3f}",
                        "disc": f"{partial_losses['disconnection'].item():.0f}",
                        "hill": (
                            f"{partial_losses['hill'].item():.2e}"
                            if partial_losses["hill"].item() > 10
                            else f"{partial_losses['hill'].item():.3f}"
                        ),
                        "io": f"{partial_losses['io'].item():.3f}",
                        "smooth": (
                            f"{partial_losses['smooth'].item():.2e}"
                            if partial_losses["smooth"].item() > 10
                            else f"{partial_losses['smooth'].item():.3f}"
                        ),
                        "density": f"{partial_losses['density'].item():.0f}",
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


def train_model_progressive_with_dropout(
        config_path: str | Path,
        save_path: Optional[str | Path] = None,
        device: str | torch.device = "cpu",
        num_cycles: int = 3,
        early_stopping_patience: int = 100,
        dropout_rate: float = 0.2,  # Parameter dropout rate
        component_dropout_rate: float = 0.1,  # Component dropout rate
        use_block_update: bool = False,  # Whether to use block coordinate descent
        block_update_cycle: int = 3,  # Number of epochs before switching parameter blocks
        min_lr: float = 5e-7,
):
    """Train network generator model using progressive loss optimization with parameter dropout.

    Args:
        config_path: Path to configuration file
        save_path: Optional path to save checkpoints
        device: Device to train on
        num_cycles: Number of training cycles
        early_stopping_patience: Patience for early stopping
        dropout_rate: Probability of dropping out individual parameters
        component_dropout_rate: Probability of dropping out entire components
        use_block_update: Whether to use block coordinate updates
        block_update_cycle: Number of epochs before switching parameter blocks

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
        "density": [],
        "dropout_applied": [],  # Track when dropout was applied
    }

    best_loss = float("inf")
    best_state_dict = None

    # Calculate total epochs for progress bar
    total_epochs = sum(phase["epochs"] for phase in phases) * num_cycles
    progress_bar = tqdm(total=total_epochs, desc="Training")
    current_epoch = 0
    current_lr = config["learning_rate"]
    lr_cycle_decay = config.get("lr_cycle_decay", 0.9)

    # Function to apply component dropout
    def apply_component_dropout(model, rate):
        """Temporarily zero out entire components during forward pass."""
        # Store original alpha values
        original_alpha = model.alpha.data.clone()

        # Create dropout mask for components
        mask = torch.rand(model.M, device=model.alpha.device) > rate

        # Apply mask to alpha parameters (zeroing out entire components)
        model.alpha.data = model.alpha.data * mask

        return original_alpha

    # Function to apply parameter dropout (create parameter masks)
    def create_param_masks(model, rate):
        """Create masks for parameter dropout."""
        masks = {}

        # Create dropout masks for each parameter
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Different dropout strategies for different parameters
                if 'alpha' in name:
                    # Use higher retention rate for alpha (fewer dropouts)
                    mask = torch.rand_like(param) > (rate * 0.5)
                else:
                    mask = torch.rand_like(param) > rate
                masks[name] = mask

        return masks

    # Function to apply parameter masks to gradients
    def apply_masks_to_gradients(model, masks):
        """Apply saved masks to parameter gradients."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in masks and param.grad is not None:
                param.grad = param.grad * masks[name]

    # Training cycles
    for cycle in range(num_cycles):

        if cycle > 0:
            current_lr = max(current_lr * lr_cycle_decay, min_lr)
            logger.info(f"Cycle {cycle}: Reducing learning rate to {current_lr:.2e}")

        for phase_idx, phase in enumerate(phases):
            # Create phase-specific config
            phase_config = config.copy()
            phase_weights = {k: 0.0 for k in config["loss_weights"].keys()}
            phase_weights.update(phase["weights"])
            phase_config["loss_weights"] = phase_weights

            # Check if density weight is zero in this phase
            density_weight = phase_weights.get("density", 0.0)
            density_training = density_weight > 0

            # Freeze/unfreeze model parameters based on density weight
            if hasattr(model, "freeze_density") and hasattr(model, "unfreeze_density"):
                if density_training:
                    model.unfreeze_density()
                    logger.info(f"Phase {phase_idx} ({phase['name']}): Unfreezing density parameters")
                else:
                    model.freeze_density()
                    logger.info(f"Phase {phase_idx} ({phase['name']}): Freezing density parameters")
            else:
                if not density_training:
                    logger.info(
                        f"Phase {phase_idx} ({phase['name']}): Density weight is zero but model doesn't support parameter freezing")

            # Adjust dropout rate based on phase
            # Earlier phases benefit from higher dropout to explore parameter space
            # Later phases need less dropout for fine-tuning
            phase_dropout_rate = dropout_rate * (1.0 - phase_idx / len(phases) * 0.5)
            phase_component_dropout = component_dropout_rate * (1.0 - phase_idx / len(phases) * 0.7)

            # Create new optimizer with only trainable parameters
            if "relative_learning_rate" in phase:
                phase_lr = current_lr * phase["relative_learning_rate"]
            else:
                # For backward compatibility, still check for absolute learning_rate
                phase_lr = phase.get("learning_rate", current_lr)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(trainable_params, lr=phase_lr)

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.75,
                patience=max(20, phase["epochs"] // 10),
                min_lr=min_lr,
                threshold=1e-4,
                threshold_mode="rel",
            )

            # For block updates, determine which parameter group to update
            if use_block_update:
                # We'll cycle through different parameter groups
                param_groups = ['alpha', 'U', 'V']
                current_group_idx = 0
                epochs_in_current_group = 0

            # Training loop for this phase
            for epoch in range(phase["epochs"]):
                # Set up block update if enabled
                if use_block_update:
                    # Check if it's time to switch parameter groups
                    if epochs_in_current_group >= block_update_cycle:
                        current_group_idx = (current_group_idx + 1) % len(param_groups)
                        epochs_in_current_group = 0

                    # Update requires_grad for each parameter based on current group
                    current_group = param_groups[current_group_idx]
                    for name, param in model.named_parameters():
                        # By default, freeze all parameters
                        param.requires_grad = False

                        # Unfreeze the current parameter group
                        if current_group == 'alpha' and 'alpha' in name:
                            param.requires_grad = True
                        elif current_group == 'U' and '.U' in name:
                            param.requires_grad = True
                        elif current_group == 'V' and '.V' in name:
                            param.requires_grad = True

                    epochs_in_current_group += 1

                # Zero gradients
                optimizer.zero_grad()

                # Apply component dropout if enabled
                apply_dropout = np.random.random() < 0.7  # Apply dropout 70% of the time
                dropout_applied = False

                if apply_dropout and phase_component_dropout > 0:
                    # Apply component dropout
                    original_alpha = apply_component_dropout(model, phase_component_dropout)
                    dropout_applied = True

                # Create parameter masks for gradient dropout if enabled
                param_masks = None
                if apply_dropout and phase_dropout_rate > 0 and not use_block_update:
                    param_masks = create_param_masks(model, phase_dropout_rate)
                    dropout_applied = True

                # Forward pass and compute loss
                log_weights = model()
                loss, partial_losses = compute_loss(log_weights, phase_config)

                # Backward pass
                loss.backward()

                # Apply parameter masks to gradients if dropout is enabled
                if param_masks is not None:
                    apply_masks_to_gradients(model, param_masks)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad and p.grad is not None],
                    phase_config.get("max_grad_norm", 1.0),
                )

                # Update weights
                optimizer.step()

                # Restore original alpha values if component dropout was applied
                if apply_dropout and phase_component_dropout > 0:
                    model.alpha.data = original_alpha

                # Reset requires_grad if block update was used
                if use_block_update:
                    for param in model.parameters():
                        param.requires_grad = True

                # Get current learning rate
                current_lr = optimizer.param_groups[0]["lr"]

                # Update history
                update_history(history, loss, partial_losses, current_lr, cycle, phase_idx)
                history["dropout_applied"].append(float(dropout_applied))

                # Track best model
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state_dict = model.state_dict().copy()

                # Learning rate scheduling
                scheduler.step(loss)

                # Update progress bar with more detailed info
                progress_bar.set_postfix(
                    {
                        "cycle": f"{cycle}/{num_cycles - 1}",
                        "phase": f"{phase_idx + 1}/{len(phases)}",
                        "loss": f"{loss.item():.3f}",
                        "corr": f"{partial_losses['correlation'].item():.3f}",
                        "disc": f"{partial_losses['disconnection'].item():.0f}",
                        "hill": (
                            f"{partial_losses['hill'].item():.2e}"
                            if partial_losses["hill"].item() > 10
                            else f"{partial_losses['hill'].item():.3f}"
                        ),
                        "io": f"{partial_losses['io'].item():.3f}",
                        "smooth": (
                            f"{partial_losses['smooth'].item():.2e}"
                            if partial_losses["smooth"].item() > 10
                            else f"{partial_losses['smooth'].item():.3f}"
                        ),
                        "density": f"{partial_losses['density'].item():.0f}",
                        "dropout": f"{'on' if dropout_applied else 'off'}",
                        "lr": f"{current_lr:.2e}",
                    }
                )
                progress_bar.update(1)
                current_epoch += 1

            current_lr = optimizer.param_groups[0]["lr"]

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