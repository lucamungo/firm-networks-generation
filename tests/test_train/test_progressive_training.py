"""Tests for the progressive training module."""

import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from network_generation.train import train_model_progressive

logger = logging.getLogger(__name__)


class TestProgressiveTraining:
    """Test suite for progressive training functionality."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path: Path):
        """Setup test configuration and paths."""
        self.config = {
            # Network size
            "N": 4,  # Tiny network
            "M": 2,  # Small number of components
            # Target correlations
            "correlation_targets": {
                "log_in_degree_out_degree": 0.7,
                "log_in_strength_out_strength": 0.8,
                "log_out_strength_out_degree": 0.6,
            },
            # Target Hill exponents
            "hill_exponent_targets": {
                "in_degree": -2.0,
                "out_degree": -2.0,
                "in_strength": -2.0,
                "out_strength": -2.0,
            },
            # Group assignments and target IO matrix
            "group_assignments": {0: [0, 1], 1: [2, 3]},  # Split nodes into 2 groups
            "io_matrix_target": [[1.0, 2.0], [2.0, 1.0]],
            # Default loss weights
            "loss_weights": {
                "correlation": 1.0,
                "hill": 1.0,
                "io": 1.0,
                "smooth": 0.1,
            },
            # Progressive training phases
            "training_phases": [
                {
                    "name": "IO Matrix",
                    "epochs": 100,
                    "weights": {
                        "correlation": 0.0,
                        "hill": 0.0,
                        "io": 1.0,
                        "smooth": 0.0,
                        "continuity": 0.0,
                    },
                },
                {
                    "name": "Correlations",
                    "epochs": 100,
                    "weights": {
                        "correlation": 1.0,
                        "hill": 0.0,
                        "io": 0.5,
                        "smooth": 0.0,
                        "continuity": 0.0,
                    },
                },
                {
                    "name": "Hill Exponents",
                    "epochs": 100,
                    "weights": {
                        "correlation": 0.5,
                        "hill": 1.0,
                        "io": 0.5,
                        "smooth": 0.0,
                        "continuity": 0.0,
                    },
                },
                {
                    "name": "Fine Tuning",
                    "epochs": 100,
                    "weights": {
                        "correlation": 1.0,
                        "hill": 1.0,
                        "io": 1.0,
                        "smooth": 0.1,
                        "continuity": 0.0,
                    },
                },
            ],
            # Training hyperparameters
            "learning_rate": 0.01,
            "beta_degree": 10.0,
            "beta_ccdf": 10.0,
            "beta_tail": 10.0,
            "tail_fraction": 0.5,
            "num_ccdf_points": 10,
            "num_epochs": 400,  # Total epochs across all phases
        }

        self.config_path = tmp_path / "tiny_progressive_config.json"
        with open(self.config_path, "w") as f:
            json.dump(self.config, f)

        self.save_path = tmp_path / "model.pt"

    def test_cycle_completion(self):
        """Test that all cycles and phases complete with correct number of epochs."""
        num_cycles = 3
        model, history = train_model_progressive(self.config_path, num_cycles=num_cycles)

        # Check total number of epochs
        total_phase_epochs = sum(phase["epochs"] for phase in self.config["training_phases"])
        expected_total_epochs = total_phase_epochs * num_cycles
        assert (
            len(history["total"]) == expected_total_epochs
        ), f"Expected {expected_total_epochs} epochs, got {len(history['total'])}"

        # Check cycle tracking
        assert (
            len(history["cycle"]) == expected_total_epochs
        ), f"Expected {expected_total_epochs} cycles, got {len(history['cycle'])}"
        assert set(history["cycle"]) == set(
            range(num_cycles)
        ), f"Expected cycles {range(num_cycles)}, got {history['cycle']}"

        # Check phase tracking
        assert (
            len(history["phase"]) == expected_total_epochs
        ), f"Expected {expected_total_epochs} phases, got {len(history['phase'])}"
        num_phases = len(self.config["training_phases"])
        assert set(history["phase"]) == set(
            range(num_phases)
        ), f"Expected phases {range(num_phases)}, got {history['phase']}"

    def test_objective_improvement(self):
        """Test that objectives improve across cycles."""
        num_cycles = 3
        model, history = train_model_progressive(self.config_path, num_cycles=num_cycles)

        # Group history by phase and cycle
        phase_lengths = [phase["epochs"] for phase in self.config["training_phases"]]
        num_phases = len(phase_lengths)
        total_epochs_per_cycle = sum(phase_lengths)

        for phase_idx in range(num_phases):
            # Get active weights for this phase
            phase_weights = self.config["training_phases"][phase_idx]["weights"]
            active_components = [k for k, v in phase_weights.items() if v > 0]

            # For each active component, check if it improves across cycles
            for component in active_components:
                if component not in history:
                    continue

                cycle_mins = []  # Track best loss per cycle for this component
                for cycle in range(num_cycles):
                    start_idx = cycle * total_epochs_per_cycle + sum(phase_lengths[:phase_idx])
                    end_idx = start_idx + phase_lengths[phase_idx]
                    cycle_losses = history[component][start_idx:end_idx]
                    cycle_mins.append(min(cycle_losses))

                # Check if best loss generally improves across cycles
                # We don't require monotonic improvement but should see some overall progress
                assert min(cycle_mins) < cycle_mins[0], (
                    f"Component {component} shows no improvement across cycles "
                    f"in phase {phase_idx}"
                )

    def test_loss_composition(self):
        """Test that total loss matches weighted sum of components."""
        model, history = train_model_progressive(self.config_path, num_cycles=2)

        # Check random points throughout training
        num_samples = 10
        total_epochs = len(history["total"])
        sample_indices = np.random.choice(total_epochs, num_samples, replace=False)

        for idx in sample_indices:
            # Get cycle and phase
            cycle = history["cycle"][idx]
            phase = history["phase"][idx]

            # Get weights for this phase
            weights = self.config["training_phases"][phase]["weights"]

            # Get component losses
            components = {
                k: history[k][idx] for k in ["correlation", "hill", "io", "smooth"] if k in history
            }

            # Compute expected total
            active_weights = {k: v for k, v in weights.items() if v > 0}
            if active_weights:
                normalized_losses = []
                for name, weight in active_weights.items():
                    loss = components[name]
                    scale = 1.0 + loss
                    normalized_losses.append(weight * loss / scale)
                expected_total = sum(normalized_losses) / sum(active_weights.values())

                # Check that actual total matches expected
                actual_total = history["total"][idx]
                assert abs(actual_total - expected_total) < 1e-5, (
                    f"Total loss mismatch at epoch {idx} " f"(cycle {cycle}, phase {phase})"
                )

    def test_save_and_load(self):
        """Test that models can be saved and loaded correctly with cycling."""
        # Train and save
        model, _ = train_model_progressive(self.config_path, save_path=self.save_path, num_cycles=2)

        # Load
        loaded_model = model.from_pretrained(self.save_path)

        # Check parameters match
        assert torch.allclose(loaded_model.alpha, model.alpha)
        assert torch.allclose(loaded_model.U, model.U)
        assert torch.allclose(loaded_model.V, model.V)

    def test_device_support(self):
        """Test training on different devices with cycling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Train on CPU
        model_cpu, history_cpu = train_model_progressive(
            self.config_path, device="cpu", num_cycles=2
        )

        # Train on CUDA
        model_cuda, history_cuda = train_model_progressive(
            self.config_path, device="cuda", num_cycles=2
        )

        # Both should reduce loss
        assert history_cpu["total"][-1] < history_cpu["total"][0]
        assert history_cuda["total"][-1] < history_cuda["total"][0]

    def test_learning_rate_decay(self):
        """Test that learning rate properly decays across cycles."""
        num_cycles = 3
        model, history = train_model_progressive(self.config_path, num_cycles=num_cycles)

        # Group learning rates by cycle
        phase_lengths = [phase["epochs"] for phase in self.config["training_phases"]]
        total_epochs_per_cycle = sum(phase_lengths)

        for cycle in range(num_cycles - 1):
            # Get average LR for current and next cycle
            current_cycle_start = cycle * total_epochs_per_cycle
            next_cycle_start = (cycle + 1) * total_epochs_per_cycle

            current_cycle_lr = np.mean(
                history["learning_rate"][current_cycle_start:next_cycle_start]
            )
            next_cycle_lr = np.mean(
                history["learning_rate"][
                    next_cycle_start : next_cycle_start + total_epochs_per_cycle
                ]
            )

            # Check that LR decreases between cycles
            assert (
                next_cycle_lr < current_cycle_lr
            ), f"Learning rate not decreasing between cycles {cycle} and {cycle + 1}"
