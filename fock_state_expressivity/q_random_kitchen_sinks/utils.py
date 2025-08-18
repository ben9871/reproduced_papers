import os
from datetime import datetime

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch


def combine_saved_figures(q_approx=True, path="./results/"):
    r_values = [1, 10, 100]
    gamma_values = list(range(1, 11))  # gamma from 1 to 10

    fig, axes = plt.subplots(len(r_values), len(gamma_values), figsize=(15, 4))

    for i, r in enumerate(r_values):
        for j, gamma in enumerate(gamma_values):
            sigma = 1.0 / gamma
            if q_approx:
                filename = f"q_rand_kitchen_sinks_R_{r}_sigma_{sigma}.png"
            else:
                filename = f"classical_rand_kitchen_sinks_R_{r}_sigma_{sigma}.png"
            filepath = os.path.join("./results/", filename)

            if os.path.exists(filepath):
                img = mpimg.imread(filepath)
                ax = axes[i, j]
                ax.imshow(img)
                ax.axis("off")  # Hide axis ticks
                if i == 0:
                    ax.set_title(f"γ = {gamma}", fontsize=10)
                if j == 0:
                    ax.text(
                        0,
                        0.5,
                        f"R = {r}",
                        fontsize=10,
                        va="center",
                        ha="right",
                        transform=ax.transAxes,
                    )
            else:
                print(f"Warning: {filepath} not found.")

    plt.tight_layout()
    # plt.subplots_adjust(left=0.01)
    plt.subplots_adjust(left=0.1, wspace=0.05, hspace=0.1)
    if q_approx:
        q_dir = os.path.join(path, "q_rand_kitchen_sinks_overall.png")
        plt.savefig(q_dir, dpi=600)
    else:
        dir = os.path.join(path, "rand_kitchen_sinks_overall.png")
        plt.savefig(dir, dpi=600)
    plt.show()
    plt.close()
    return


def save_hyperparameters(hp, filepath="./results/"):
    """
    Save hyperparameters to a text file in a visually pleasing format.

    Args:
        hp: Hyperparameters object
        filepath: Path where to save the file
    """
    # Create results directory if it doesn't exist
    dir = os.path.join(filepath, "q_rand_kitchen_sinks_hps.txt")
    os.makedirs(os.path.dirname(dir), exist_ok=True)

    with open(dir, "w") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("QUANTUM RANDOM KITCHEN SINKS HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")

        # Data Parameters
        f.write("📊 DATA PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of samples:       {hp.n_samples:>10}\n")
        f.write(f"Noise level:             {hp.noise:>10}\n")
        f.write(f"Random state:            {hp.random_state:>10}\n")
        f.write(f"Scaling method:          {hp.scaling:>10}\n")
        f.write(f"Test proportion:         {hp.test_prop:>10.2f}\n")
        f.write("\n")

        # Quantum Parameters
        f.write("⚛️  QUANTUM PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Number of photons:       {hp.num_photon:>10}\n")
        f.write(f"Output mapping strategy: {hp.output_mapping_strategy:>10}\n")
        f.write(f"No bunching:             {str(hp.no_bunching):>10}\n")
        f.write(f"Circuit type:            {hp.circuit:>10}\n")
        f.write(f"Pre-encoding scaling:    {hp.pre_encoding_scaling:>10.6f}\n")

        # Handle z_matrix_scaling display
        if isinstance(hp.z_q_matrix_scaling, str):
            z_val = hp.z_q_matrix_scaling
        else:
            z_val = f"{hp.z_q_matrix_scaling:>10.6f}"
        f.write(f"Z Q matrix scaling:      {z_val}\n")
        f.write(f"Train hybrid model:      {str(hp.train_hybrid_model):>10}\n")
        f.write(f"Hybrid model data:      {str(hp.hybrid_model_data):>10}\n")
        f.write("\n")

        # Training Parameters
        f.write("🏋️  TRAINING PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Batch size:              {hp.batch_size:>10}\n")
        f.write(f"Optimizer:               {hp.optimizer:>10}\n")
        f.write(f"Learning rate:           {hp.learning_rate:>10.6f}\n")
        f.write(f"Betas:                   {str(hp.betas):>10}\n")
        f.write(f"Weight decay:            {hp.weight_decay:>10.6f}\n")
        f.write(f"Number of epochs:        {hp.num_epochs:>10}\n")
        f.write(f"C parameter:             {hp.C:>10.2f}\n")
        f.write("\n")

        # Model Parameters
        f.write("🔧 MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"r parameter:             {hp.r:>10}\n")
        f.write(f"Gamma parameter:         {hp.gamma:>10}\n")
        f.write("\n")

        # Random Parameters
        f.write("🎲 RANDOM PARAMETERS\n")
        f.write("-" * 40 + "\n")
        if hp.w is not None:
            if isinstance(hp.w, torch.Tensor):
                w_shape = tuple(hp.w.shape)
                w_type = "Tensor"
            elif isinstance(hp.w, np.ndarray):
                w_shape = hp.w.shape
                w_type = "Array"
            else:
                w_shape = "Unknown"
                w_type = str(type(hp.w).__name__)
            f.write(f"Weights (w):             {w_type} {w_shape}\n")
        else:
            f.write(f"Weights (w):             {'None':>10}\n")

        if hp.b is not None:
            if isinstance(hp.b, torch.Tensor):
                b_shape = tuple(hp.b.shape)
                b_type = "Tensor"
            elif isinstance(hp.b, np.ndarray):
                b_shape = hp.b.shape
                b_type = "Array"
            else:
                b_shape = "Unknown"
                b_type = str(type(hp.b).__name__)
            f.write(f"Biases (b):              {b_type} {b_shape}\n")
        else:
            f.write(f"Biases (b):              {'None':>10}\n")

        # Footer
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of hyperparameters file\n")
        f.write("=" * 80 + "\n")

    print(f"Hyperparameters saved to: {dir}")
    return


def create_experiment_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{timestamp}"
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir
