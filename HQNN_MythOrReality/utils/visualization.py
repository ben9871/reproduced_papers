"""Visualisation helpers for inspecting learned parameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import torch


def visualize_scale_parameters(scale_layer: torch.nn.Module) -> None:
    """Plot the learned scale parameters and save to disk."""
    if not hasattr(scale_layer, "scale"):
        raise AttributeError("Scale layer missing expected `scale` attribute.")

    scale_data = scale_layer.scale.data.detach().cpu().numpy()

    if scale_data.size == 1:
        print(f"Learned scale parameter: {scale_data.item():.4f}")
        return

    flattened = scale_data.reshape(-1)
    fig, (ax_bar, ax_heatmap) = plt.subplots(2, 1, figsize=(10, 6))

    ax_bar.bar(range(len(flattened)), flattened)
    ax_bar.set_title("Learned Scale Parameters")
    ax_bar.set_xlabel("Parameter Index")
    ax_bar.set_ylabel("Value")

    sns.heatmap(
        flattened.reshape(1, -1),
        cmap="viridis",
        annot=len(flattened) < 20,
        ax=ax_heatmap,
    )
    ax_heatmap.set_title("Scale Parameters Heatmap")
    ax_heatmap.set_xlabel("Parameter Index")

    plt.tight_layout()
    plt.savefig("scale_parameters.png")
    plt.show()
