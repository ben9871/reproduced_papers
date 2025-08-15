import datetime
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        """
        Corrected InfoNCE implementation.

        Args:
            x1: First augmented views, shape (batch_size, feature_dim)
            x2: Second augmented views, shape (batch_size, feature_dim)
        """
        # Normalize features for better stability
        x1 = f.normalize(x1, dim=1)
        x2 = f.normalize(x2, dim=1)

        batch_size = x1.shape[0]
        features = torch.cat([x1, x2], dim=0)  # Shape: (2*batch_size, feature_dim)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create correct labels for positive pairs
        # For sample i in first half: positive is sample i in second half (index i+batch_size)
        # For sample i in second half: positive is sample i-batch_size in first half
        labels = torch.cat(
            [
                torch.arange(
                    batch_size, 2 * batch_size
                ),  # [batch_size, ..., 2*batch_size-1]
                torch.arange(0, batch_size),  # [0, 1, ..., batch_size-1]
            ]
        ).to(features.device)

        # Remove self-similarities by masking diagonal
        mask = torch.eye(2 * batch_size, dtype=bool, device=features.device)
        similarity_matrix.masked_fill_(mask, -9e15)

        # Compute InfoNCE loss using cross-entropy
        loss = f.cross_entropy(similarity_matrix, labels)

        return loss


def training_step(model, train_loader, optimizer):
    pbar = tqdm(train_loader)
    total_loss = 0.0
    for (x1, x2), _target in pbar:
        loss = model(x1, x2)

        # Check for NaN/inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss detected: {loss}")
            continue

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

    return total_loss / len(train_loader)


def get_results_dir(args):
    """Create and return results directory path based on training type"""
    if args.merlin:
        base_dir = "results/merlin"
    elif args.qiskit:
        base_dir = "results/qiskit"
    else:
        base_dir = "results/classical"

    # Create datetime subdirectory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, timestamp)
    os.makedirs(results_dir, exist_ok=True)

    return results_dir


def save_metrics_during_training(
    results_dir,
    epoch,
    ssl_loss=None,
    train_loss=None,
    val_loss=None,
    train_acc=None,
    val_acc=None,
):
    """Save metrics to JSON file during training"""
    metrics_file = os.path.join(results_dir, "training_metrics.json")

    # Load existing metrics or create new
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {
            "ssl_training_losses": [],
            "linear_evaluation": {
                "train_losses": [],
                "val_losses": [],
                "train_accuracies": [],
                "val_accuracies": [],
            },
        }

    # Update metrics
    if ssl_loss is not None:
        metrics["ssl_training_losses"].append({"epoch": epoch, "loss": ssl_loss})

    if train_loss is not None:
        metrics["linear_evaluation"]["train_losses"].append(
            {"epoch": epoch, "loss": train_loss}
        )

    if val_loss is not None:
        metrics["linear_evaluation"]["val_losses"].append(
            {"epoch": epoch, "loss": val_loss}
        )

    if train_acc is not None:
        metrics["linear_evaluation"]["train_accuracies"].append(
            {"epoch": epoch, "accuracy": train_acc}
        )

    if val_acc is not None:
        metrics["linear_evaluation"]["val_accuracies"].append(
            {"epoch": epoch, "accuracy": val_acc}
        )

    # Save updated metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


def train(model, train_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    training_losses = []

    # Create results directory
    results_dir = get_results_dir(args)
    print(f"Saving training results to: {results_dir}")

    for epoch in range(args.epochs):
        loss = training_step(model, train_loader, optimizer)
        print(f"epoch: {epoch + 1}/{args.epochs}, training loss: {loss}")
        training_losses.append(loss)

        # Save SSL training loss during training
        save_metrics_during_training(results_dir, epoch + 1, ssl_loss=loss)

    return model, training_losses, results_dir


def linear_evaluation(model, train_loader, val_loader, args, results_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    for epoch in range(args.le_epochs):
        # training
        model.train()
        pbar = tqdm(train_loader)
        train_acc = 0
        train_loss_total = 0
        for img, target in pbar:
            output = model(img)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            accuracy = (predicted == target).sum().item()
            train_acc += accuracy
            train_loss_total += loss.item()
            pbar.set_postfix(
                {
                    "Training Loss": f"{loss.item():.4f} - Training Accuracy: {accuracy:.4f}"
                }
            )

        # validation
        model.eval()
        pbar = tqdm(val_loader)
        val_acc = 0
        val_loss_total = 0
        with torch.no_grad():
            for img, target in pbar:
                output = model(img)
                loss = criterion(output, target)
                _, predicted = torch.max(output.data, 1)
                accuracy = (predicted == target).sum().item()
                val_acc += accuracy
                val_loss_total += loss.item()
                pbar.set_postfix(
                    {
                        "Validation Loss": f"{loss.item():.4f} - Validation Accuracy: {accuracy:.4f}"
                    }
                )

        avg_train_acc = train_acc / len(train_loader.dataset)
        avg_val_acc = val_acc / len(val_loader.dataset)
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(avg_val_acc)

        # Save metrics during training
        save_metrics_during_training(
            results_dir,
            epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=avg_train_acc,
            val_acc=avg_val_acc,
        )

        print(
            f"Epoch {epoch + 1}/{args.le_epochs}: Train Acc = {avg_train_acc:.4f}, Val Acc = {avg_val_acc:.4f}"
        )

    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_loss(training_losses, args):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, args.epochs + 1),
        training_losses,
        "b-",
        linewidth=2,
        label="Training Loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title = "Classical"
    if args.merlin:
        title = "Quantum_MerLin"
    if args.qiskit:
        title = "Quantum_Qiskit"
    plt.title(f"SSL Training Loss ({title} Network)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"ssl_training_loss_{title}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_evaluation_metrics(train_losses, val_losses, train_accs, val_accs, args):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))
    title = "Classical"
    if args.merlin:
        title = "Quantum_MerLin"
    if args.qiskit:
        title = "Quantum_Qiskit"
    # Plot losses
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Linear Evaluation Losses ({title} Network)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", linewidth=2, label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", linewidth=2, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Linear Evaluation Accuracies ({title} Network)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f"evaluation_metrics_{title}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def save_results_to_json(
    args,
    ssl_training_losses,
    ft_train_losses,
    ft_val_losses,
    ft_train_accs,
    ft_val_accs,
    results_dir,
):
    # Determine title based on quantum mode
    title = "Classical"
    if args.merlin:
        title = "Quantum_MerLin"
    if args.qiskit:
        title = "Quantum_Qiskit"

    # Save final summary to results directory
    summary_file = os.path.join(results_dir, "experiment_summary.json")

    # Create experiment entry
    experiment = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_type": title,
        "arguments": {
            "quantum-MerLin": args.merlin,
            "quantum-Qiskit": args.qiskit,
            "epochs": args.epochs,
            "le_epochs": args.le_epochs,
            "batch_size": args.batch_size,
            "classes": args.classes,
            "width": args.width,
            "loss_dim": args.loss_dim,
            "temperature": args.temperature,
            "modes": getattr(args, "modes", None),
            "no_bunching": getattr(args, "no_bunching", None),
            "datadir": args.datadir,
        },
        "ssl_training_losses": ssl_training_losses,
        "linear_evaluation": {
            "train_losses": ft_train_losses,
            "val_losses": ft_val_losses,
            "train_accuracies": ft_train_accs,
            "val_accuracies": ft_val_accs,
            "final_val_accuracy": ft_val_accs[-1] if ft_val_accs else 0.0,
            "best_val_accuracy": max(ft_val_accs) if ft_val_accs else 0.0,
        },
    }

    # Save experiment summary to results directory
    with open(summary_file, "w") as f:
        json.dump(experiment, f, indent=2)

    # Also save to the original location for backwards compatibility
    filename = f"{title}_results.json"
    try:
        with open(filename) as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    # Append new experiment
    results.append(experiment)

    # Save updated results
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to:")
    print(f"  - {summary_file}")
    print(f"  - {filename}")
    print(f"  - Training metrics: {os.path.join(results_dir, 'training_metrics.json')}")
    print(
        f"Final validation accuracy: {experiment['linear_evaluation']['final_val_accuracy']:.4f}"
    )
    print(
        f"Best validation accuracy: {experiment['linear_evaluation']['best_val_accuracy']:.4f}"
    )
