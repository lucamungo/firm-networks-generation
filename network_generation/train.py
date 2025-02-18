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

    # Initialize loss history with all possible loss names
    history = {
        "total": [],
        # Main loss components
        "correlation": [],
        "hill": [],
        "io": [],
        "smooth": [],
        # Individual correlation losses
        "correlation_log_in_strength_out_strength": [],
        "correlation_log_in_degree_out_degree": [],
        "correlation_log_out_strength_out_degree": [],
        # Individual hill losses
        "hill_in_degree": [],
        "hill_out_degree": [],
        "hill_in_strength": [],
        "hill_out_strength": [],
        # Individual smoothness losses
        "smooth_in_degree": [],
        "smooth_out_degree": [],
        "smooth_in_strength": [],
        "smooth_out_strength": [],
    }

    # Training loop
    logger.info("Starting training...")
    progress_bar = tqdm(range(config["num_epochs"]), desc="Training")
    for epoch in progress_bar:
        optimizer.zero_grad()

        # Forward pass
        log_weights = model()

        # Compute loss
        loss, partial_losses = compute_loss(log_weights, config)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Record history
        history["total"].append(loss.item())
        for name, value in partial_losses.items():
            if name in history:  # Only record if we initialized the history for it
                history[name].append(value.item())

        # Update progress bar
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "corr": f"{partial_losses['correlation'].item():.4f}",
                "hill": f"{partial_losses['hill'].item():.4f}",
                "io": f"{partial_losses['io'].item():.4f}",
                "smooth": f"{partial_losses['smooth'].item():.4f}",
            }
        )

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{config['num_epochs']}: "
                f"loss={loss.item():.4f}, "
                f"correlation={partial_losses['correlation'].item():.4f}, "
                f"hill={partial_losses['hill'].item():.4f}, "
                f"io={partial_losses['io'].item():.4f}, "
                f"smooth={partial_losses['smooth'].item():.4f}"
            )

    # Save model if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
            },
            save_path,
        )
        logger.info(f"Saved model to {save_path}")

    return model, history
