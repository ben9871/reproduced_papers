"""Training helpers for HQNN experiments."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from tqdm import tqdm


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters of a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    num_epochs: int = 25,
    lr: float = 0.01,
    device: torch.device | str = "cpu",
) -> tuple[list[float], list[float], float, list[float], list[float]]:
    """Train the provided model and track accuracy/loss curves."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.999))

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accuracies: list[float] = []
    val_accuracies: list[float] = []
    best_val_acc = 0.0

    progress_bar = tqdm(range(num_epochs))
    for _epoch in progress_bar:
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.inference_mode():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = total_val_loss / max(len(val_loader), 1)
        train_acc = 100 * correct_train / max(total_train, 1)
        val_acc = 100 * correct_val / max(total_val, 1)
        best_val_acc = max(best_val_acc, val_acc)

        progress_bar.set_postfix(
            {"val_loss": f"{avg_val_loss:.4f}", "val_acc": f"{val_acc:.2f}"}
        )

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

    return train_losses, val_losses, best_val_acc, train_accuracies, val_accuracies
