"""Dataset loading utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from merlin.datasets import spiral
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class SpiralDatasetConfig:
    """Configuration values for the synthetic spiral dataset."""

    num_instances: int
    num_features: int
    num_classes: int
    test_size: float = 0.2
    random_state: int = 42


def load_spiral_dataset(
    config: SpiralDatasetConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Load and normalise the synthetic spiral dataset."""

    x, y, _ = spiral.get_data(
        num_instances=config.num_instances,
        num_features=config.num_features,
        num_classes=config.num_classes,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (
        x_train_tensor,
        x_test_tensor,
        y_train_tensor,
        y_test_tensor,
        config.num_features,
        config.num_classes,
    )
