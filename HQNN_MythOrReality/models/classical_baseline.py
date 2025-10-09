"""Classical MLP baseline utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.data import SpiralDatasetConfig, load_spiral_dataset
from utils.training import count_parameters, train_model


class MLP(nn.Module):
    """Flexible MLP that supports a variable number of hidden dimensions."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim

        if not hidden_dims:
            layers.append(nn.Linear(prev_dim, output_dim))
        else:
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for the classical baseline sweep."""

    nb_features: int
    hidden_dims: Sequence[int]
    nb_classes: int
    nb_samples: int
    repetitions: int
    lr: float
    batch_size: int


def generate_mlp_architectures(max_hidden_layers: int = 4) -> Iterable[Sequence[int]]:
    """Generate MLP hidden dimension configurations."""
    if max_hidden_layers < 1:
        return []
    base_dims = [4, 8, 16, 32, 64, 128]
    yield tuple()
    for depth in range(1, max_hidden_layers + 1):
        for base in base_dims:
            dims = [base // (2**i) for i in range(depth)]
            dims = [max(dim, 1) for dim in dims]
            yield tuple(dims)


def evaluate_architecture(cfg: BaselineConfig) -> Tuple[float, float]:
    """Train and evaluate the MLP baseline for the provided configuration."""

    accuracies: List[float] = []
    for repetition in range(cfg.repetitions):
        dataset_cfg = SpiralDatasetConfig(
            num_instances=cfg.nb_samples,
            num_features=cfg.nb_features,
            num_classes=cfg.nb_classes,
            random_state=42 + repetition,
        )
        x_train, x_val, y_train, y_val, _, _ = load_spiral_dataset(dataset_cfg)

        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        model = MLP(cfg.nb_features, cfg.hidden_dims, cfg.nb_classes)
        _, _, best_acc, _, _ = train_model(
            model,
            train_loader,
            val_loader,
            num_epochs=25,
            lr=cfg.lr,
        )
        accuracies.append(best_acc)

    return float(np.mean(accuracies)), float(np.std(accuracies))
