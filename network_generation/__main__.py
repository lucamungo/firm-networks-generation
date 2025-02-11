"""Main script for network generation training."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from macrocosm_visual.viz_setup import setup_matplotlib

from network_generation.config import TrainingConfig
from network_generation.data import FirmDistribution, IOData
from network_generation.models import NetworkGenerator
from network_generation.trainer import NetworkTrainer

DEFAULT_DATA_PATH = Path(__file__).parents[1] / "data" / "national_io_tables_history"
CONFIGS_FILE = Path(__file__).parent / "default-configs.yml"

setup_matplotlib()


def load_data(config: TrainingConfig) -> torch.Tensor:
    """Load input-output table data."""
    # Use new IOData class
    io_data = IOData(io_path=config.data_path, country=config.country, year=config.year)
    return io_data.tensor


def setup_model(config: TrainingConfig, io_data: torch.Tensor) -> NetworkGenerator:
    """Set up model with computed firm distribution."""
    # Set up the FirmDistribution class
    firm_distribution = FirmDistribution(num_firms=config.num_firms, io_data=io_data)

    # Initialize model with the new firm distribution
    return NetworkGenerator(
        num_firms=config.num_firms,
        num_of_losses=8,
        targets=config.targets,
        io_data=io_data,
        firm_distribution=firm_distribution.indices,  # Use the indices property
        loss_bools=config.training.losses,
    )


def parse_args():
    """Parse command line arguments."""
    today = pd.to_datetime("today").strftime("%Y%m%d")

    parser = ArgumentParser(description="Train network generation model")
    parser.add_argument(
        "--configs", type=str, default=CONFIGS_FILE, help="Path to configuration file"
    )
    parser.add_argument("--num_firms", type=int, default=1_000, help="Number of firms to simulate")
    parser.add_argument("--epochs", type=int, default=1_000, help="Number of training epochs")
    parser.add_argument(
        "--learning_rate_val", type=float, default=1.0, help="Initial learning rate"
    )
    parser.add_argument("--country", type=str, default="FRA", help="Country code for IO tables")
    parser.add_argument(
        "--data_path", type=str, default=DEFAULT_DATA_PATH, help="Path to IO tables"
    )
    parser.add_argument("--year", type=int, default=2015, help="Year for IO tables")
    parser.add_argument(
        "--output", type=str, default=f"./results-{today}.pkl", help="Output file path"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def main():
    """Main execution function. The function parses the configs, loads the data, sets up the model, and trains it."""
    # Parse arguments and create config
    args = parse_args()
    config = TrainingConfig.from_args(args)

    # Load data and setup model
    io_data = load_data(config)
    model = setup_model(config, io_data)

    # Train model
    trainer = NetworkTrainer(model, config)
    trainer.train()


if __name__ == "__main__":
    main()
