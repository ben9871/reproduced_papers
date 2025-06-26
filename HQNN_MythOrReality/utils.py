import json
import os

import matplotlib.pyplot as plt
import numpy as np
import perceval as pcvl
import seaborn as sns
import torch
from merlin.datasets import spiral
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from tqdm import tqdm

############
### data ###
############


def load_spiral_dataset(nb_features=3, samples=5000, nb_classes=3):
    x, y, md = spiral.get_data(
        num_instances=samples,
        num_features=nb_features,
        num_classes=nb_classes,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(
        y_train, dtype=torch.long
    )  # Use long for classification targets
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    # print("giving",x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, x_test, y_train, y_test, nb_features, nb_classes


#####################
### Quantum Layer ###
#####################


def create_quantum_circuit(m, size=400):
    """Create quantum circuit with specified number of modes

    Args:
        m (int): number of modes
        size (int): size of the input data
        frequency (int): frequency of the repetition of the {encoding layers with input data in phase shifters; trainable generic interferometer}
    """

    # first trainable generic interferometer
    # here, we train both beam splitters and phase shifters
    wl = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS(theta=pcvl.P(f"bs_1_{i}"))
        // pcvl.PS(pcvl.P(f"ps_1_{i}"))
        // pcvl.BS(theta=pcvl.P(f"bs_2_{i}"))
        // pcvl.PS(pcvl.P(f"ps_2_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c = pcvl.Circuit(m)
    c.add(0, wl, merge=True)

    # encoding layers with input data in phase shifters
    c_var = pcvl.Circuit(m)
    for i in range(size):
        px = pcvl.P(f"px-{i + 1}")
        c_var.add(i % m, pcvl.PS(px))
    c.add(0, c_var, merge=True)

    # second trainable generic interferometer
    # here, we only train the phase shifters
    wr = pcvl.GenericInterferometer(
        m,
        lambda i: pcvl.BS()
        // pcvl.PS(pcvl.P(f"ps_3_{i}"))
        // pcvl.BS()
        // pcvl.PS(pcvl.P(f"ps_4_{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )

    c.add(0, wr, merge=True)

    return c


# count parameters of a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# this Module multiplies the input by a given or "learned" parameter


class ScaleLayer(nn.Module):
    def __init__(self, dim, scale_type="learned"):
        super().__init__()
        # Create a single learnable parameter (initialized to 1.0 by default)
        if scale_type == "learned":
            self.scale = nn.Parameter(torch.rand(dim))
        elif scale_type == "2pi":
            self.scale = torch.full((dim,), 2 * torch.pi)
        elif scale_type == "pi":
            self.scale = torch.full((dim,), torch.pi)
        elif scale_type == "1":
            self.scale = torch.full((dim,), 1)

    def forward(self, x):
        # Element-wise multiplication of each input element by the learned scale
        return x * self.scale


#####################
### training loop ###
#####################


def train_model(model, train_loader, val_loader, num_epochs=25, lr=0.01, device="cpu"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.8, 0.999))
    all_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_val_accuracies = []
    best_val_acc = 0
    progress_bar = tqdm(range(num_epochs))
    for _epoch in progress_bar:
        model.train()
        total_loss = 0
        total_test_loss = 0
        correct = 0
        total = 0
        train_acc = 0
        for batch_x, batch_y in train_loader:
            # Forward pass
            batch_x = batch_x.to(device)

            outputs = model(batch_x.squeeze(0).float())
            loss = criterion(outputs, batch_y.to(device))
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y.to(device)).sum().item()

            # Update tqdm with current metrics
            current_accuracy = 100 * correct / total
            train_acc += current_accuracy
        model.eval()
        correct_test = 0
        total_test = 0
        val_acc = 0
        # for batch_x, batch_y in val_loader:
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x.squeeze(0).float())
            test_loss = criterion(outputs, batch_y.to(device))  # .view(-1, 1).float()
            total_test_loss += test_loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_test += batch_y.size(0)
            correct_test += (predicted == batch_y.to(device)).sum().item()

            current_test_accuracy = 100 * correct_test / total_test
            val_acc += current_test_accuracy
        val_acc_epoch = val_acc / len(val_loader)
        train_acc_epoch = train_acc / len(train_loader)
        if val_acc_epoch > best_val_acc:
            best_val_acc = val_acc_epoch
        # Print progress
        avg_loss = total_loss / len(train_loader)
        avg_test_loss = total_test_loss / len(val_loader)

        all_losses.append(avg_loss)
        all_test_losses.append(avg_test_loss)
        all_train_accuracies.append(train_acc_epoch)
        all_val_accuracies.append(val_acc_epoch)
        progress_bar.set_postfix(
            {
                "Test Loss": f"{avg_test_loss:.4f}",
                "Test Accuracy": f"{val_acc_epoch:.4f}",
            }
        )
    """print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {avg_loss:.4f}, Test loss: {avg_test_loss:.4f}, Train acc: {current_accuracy:.4f}, Test acc: {current_test_accuracy:.4f}, Best val acc: {best_val_acc:.4f}')"""

    return (
        all_losses,
        all_test_losses,
        best_val_acc,
        all_train_accuracies,
        all_val_accuracies,
    )


#######################
### Other functions ###
#######################

# visualize the parameters of the scale layer (useful when parameters are learned)


def visualize_scale_parameters(scale_layer):
    # Get the scale parameter data as a numpy array
    scale_data = scale_layer.scale.data.cpu().numpy()

    # For a single scale parameter
    if scale_data.size == 1:
        print(f"Learned scale parameter: {scale_data.item():.4f}")

    # For a 1D array of parameters (e.g., per feature)
    elif len(scale_data.shape) == 1 or (
        len(scale_data.shape) > 1 and np.prod(scale_data.shape) == max(scale_data.shape)
    ):
        # Reshape to 1D if necessary
        scale_data = scale_data.flatten()

        plt.figure(figsize=(10, 6))

        # Option 1: Bar plot
        plt.subplot(2, 1, 1)
        plt.bar(range(len(scale_data)), scale_data)
        plt.title("Learned Scale Parameters")
        plt.xlabel("Parameter Index")
        plt.ylabel("Value")

        # Option 2: Heatmap (1D version)
        plt.subplot(2, 1, 2)
        sns.heatmap(
            scale_data.reshape(1, -1),
            cmap="viridis",
            annot=True if len(scale_data) < 20 else False,
        )
        plt.title("Scale Parameters Heatmap")
        plt.xlabel("Parameter Index")

        plt.tight_layout()
        plt.savefig("scale_parameters.png")
        plt.show()


def save_experiment_results(results, filename="lr_exp.json"):
    """
    Append experiment results to a JSON file.

    Args:
        results (dict): Dictionary containing experiment results (with float values)
        filename (str): Path to the JSON file to store results
    """
    filename = os.path.join("./results", filename)
    # Check if file exists and load existing data
    if os.path.exists(filename):
        try:
            with open(filename) as file:
                all_results = json.load(file)
        except json.JSONDecodeError:
            # Handle case where file exists but is empty or corrupted
            all_results = []
    else:
        all_results = []

    # Append new results
    all_results.append(results)

    # Write updated data back to file
    with open(filename, "w") as file:
        json.dump(all_results, file, indent=4)

    return len(all_results)
