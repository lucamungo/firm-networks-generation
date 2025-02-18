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

    # Create scheduler - reduce LR by factor of 0.5 if loss doesn't improve for 50 epochs
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
        "learning_rate": [],  # Track learning rate changes
    }

    # Keep track of best model
    best_loss = float("inf")
    best_state_dict = None

    # Training loop
    logger.info("Starting training...")
    progress_bar = tqdm(range(config["num_epochs"]), desc="Training")

    max_grad_norm = config.get("max_grad_norm", 1.0)

    for epoch in progress_bar:
        optimizer.zero_grad()

        # Forward pass
        log_weights = model()
        loss, partial_losses = compute_loss(log_weights, config)

        # Backward pass and optimize
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

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

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config['num_epochs']}: "
                f"loss={loss.item():.4f}, "
                f"lr={current_lr:.2e}, "
                f"correlation={partial_losses['correlation'].item():.4f}, "
                f"hill={partial_losses['hill'].item():.4f}, "
                f"io={partial_losses['io'].item():.4f}, "
                f"smooth={partial_losses['smooth'].item():.4f}"
            )

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best model with loss {best_loss:.4f}")

    # Save model if requested
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
        logger.info(f"Saved model to {save_path}")

    return model, history
