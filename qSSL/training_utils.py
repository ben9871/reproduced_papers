import datetime
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Args:
            features: Tensor of shape (2 * batch_size, feature_dim).
                      The first half are augmented views of instances in the batch.
                      The second half are corresponding positive pairs.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = features.shape[0] // 2
        # create pseudo labels
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (
            (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
        )
        # sim(z_i,z_j)/tau
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # mask to remove self-comparisons
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))
        # print(f"\n Similarity matrix:\n{similarity_matrix}")
        # exp(sim(z_i,z_j)/tau)
        sim_exp = torch.exp(similarity_matrix)
        # exp(sim(z_i,z_j)/tau) - same indices
        sim_exp_sum = sim_exp.sum(dim=1, keepdim=True) - torch.exp(
            similarity_matrix.diagonal().view(-1, 1)
        )
        # - log( # exp(sim(z_i,z_j)/tau) / sum )
        log_prob = similarity_matrix - torch.log(sim_exp_sum + 1e-8)

        pos_indices = torch.arange(batch_size).to(features.device)
        pos_pairs = torch.cat([pos_indices + batch_size, pos_indices]).to(
            features.device
        )
        # Apply softmax to get probabilities
        loss = -log_prob[torch.arange(2 * batch_size), pos_pairs].mean()
        """loss = -torch.log(
            F.softmax(similarity_matrix, dim=1) * labels
        ).sum(dim=1) / labels.sum(dim=1)"""
        # print(f"Loss: {loss}")
        return loss.mean()


def training_step(model, train_loader, optimizer):
    pbar = tqdm(train_loader)
    total_loss = 0.0
    iter = 0
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
        iter+=1
        if iter>3:
            return total_loss / (iter + 1)
    return total_loss / len(train_loader)


def train(model, train_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    training_losses = []
    for epoch in range(args.epochs):
        loss = training_step(model, train_loader, optimizer)
        print(f"epoch: {epoch+1}/{args.epochs}, training loss: {loss}")
        training_losses.append(loss)
    return model, training_losses


def linear_evaluation(model, train_loader, val_loader, args):
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

        print(
            f"Epoch {epoch+1}/{args.le_epochs}: Train Acc = {avg_train_acc:.4f}, Val Acc = {avg_val_acc:.4f}"
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
    plt.title(
        f'SSL Training Loss ({"Quantum" if args.quantum else "Classical"} Network)'
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f'ssl_training_loss_{"quantum" if args.quantum else "classical"}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_evaluation_metrics(train_losses, val_losses, train_accs, val_accs, args):
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot losses
    epochs = range(1, args.epochs + 1)
    ax1.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")
    ax1.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(
        f'Linear Evaluation Losses ({"Quantum" if args.quantum else "Classical"} Network)'
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot accuracies
    ax2.plot(epochs, train_accs, "b-", linewidth=2, label="Training Accuracy")
    ax2.plot(epochs, val_accs, "r-", linewidth=2, label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(
        f'Linear Evaluation Accuracies ({"Quantum" if args.quantum else "Classical"} Network)'
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(
        f'evaluation_metrics_{"quantum" if args.quantum else "classical"}.png',
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
):
    # Determine filename based on quantum mode
    filename = f'{"quantum" if args.quantum else "classical"}_results.json'

    # Create experiment entry
    experiment = {
        "timestamp": datetime.datetime.now().isoformat(),
        "arguments": {
            "quantum": args.quantum,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "classes": args.classes,
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

    # Load existing results or create new list
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

    print(f"\nResults saved to {filename}")
    print(
        f"Final validation accuracy: {experiment['linear_evaluation']['final_val_accuracy']:.4f}"
    )
    print(
        f"Best validation accuracy: {experiment['linear_evaluation']['best_val_accuracy']:.4f}"
    )