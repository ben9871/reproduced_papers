# here, we reproduce the classical baseline of the work: https://arxiv.org/abs/2412.04991


import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    count_parameters,
    load_spiral_dataset,
    save_experiment_results,
    train_model,
)


# Module to generate a MLP base on given input dims, hidden dimension and output dimension
class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) neural network implementation.
    A flexible MLP that can handle any number of hidden layers with ReLU activation.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        """
        Initialize the MLP.

        Args:
            input_dim (int): Number of input features
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Number of output features
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        # only one layer input to output
        if hidden_dims == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        # more than one hidden layer
        else:
            # build hidden layers with ReLU activation
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim

            # Output layer
            layers.append(nn.Linear(prev_dim, output_dim))
        # combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # forward pass through the network
        return self.network(x)

    def count_parameters(self):
        # count the total number of trainable parameters in the model
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def generate_architectures_from_paper():
    """Generate different MLP architectures with maximum 3 layers and neurons from {2, 4, 6, 8, 10}"""
    architectures = []

    # Neuron counts to choose from
    neuron_counts = [2, 4, 6, 8, 10]

    # 1 layer (just output layer - effectively linear model)
    architectures.append((1, []))

    # 2 layers (1 hidden + output)
    for h1 in neuron_counts:
        architectures.append((2, [h1]))

    # 3 layers (2 hidden + output)
    for h1 in neuron_counts:
        for h2 in neuron_counts:
            architectures.append((3, [h1, h2]))

    # 4 layers (3 hidden + output)
    for h1 in neuron_counts:
        for h2 in neuron_counts:
            for h3 in neuron_counts:
                architectures.append((4, [h1, h2, h3]))

    return architectures


def generate_architectures():
    """Generate different MLP architectures varying depth and width"""
    architectures = []
    architectures.append((1, 1))
    # Hidden dimensions to try
    hidden_dims = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for num_layers in range(2, 6):  # 2, 3, 4,5 layers
        for base_dim in hidden_dims:
            if num_layers == 2:
                # 2 layers: [hidden, output]
                arch = [base_dim]
            elif num_layers == 3:
                # 3 layers: [hidden1, hidden2, output]
                arch = [base_dim, base_dim // 2]
            else:  # num_layers == 4
                # 4 layers: [hidden1, hidden2, hidden3, output]
                arch = [base_dim, base_dim // 2, base_dim // 4]

            architectures.append((num_layers, arch))

    return architectures


def evaluate_architecture(
    nb_features, hidden_dims, nb_classes, nb_samples, repetitions, lr, bs
):
    """Evaluate a single architecture across multiple repetitions."""
    all_accs = []

    for _iter in range(repetitions):
        # Create model and data
        x_train, x_val, y_train, y_val, input_size, output_features = (
            load_spiral_dataset(
                nb_features=nb_features,
                samples=nb_samples,
                nb_classes=nb_classes,
            )
        )

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=bs)

        model = MLP(nb_features, hidden_dims, output_features)
        print(f"This model has {count_parameters(model)} trainable parameters")

        (cl_train_losses, cl_val_losses, best_cl_acc, cl_train_accs, cl_val_accs) = (
            train_model(model, train_loader, val_loader, num_epochs=25, lr=lr)
        )

        all_accs.append(best_cl_acc)
        print(f"Best accuracy = {best_cl_acc}")

    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)

    return mean_acc, std_acc, model


# Early stopping on the architectures with a generator
def architecture_search_generator(nb_features, architectures, nb_classes):
    """Generator that yields architectures in order of parameter count."""
    arch_with_params = []
    for num_layers, hidden_dims in architectures:
        temp_model = MLP(nb_features, hidden_dims, nb_classes)
        param_count = temp_model.count_parameters()
        arch_with_params.append((param_count, num_layers, hidden_dims))

    # Sort and yield in order
    arch_with_params.sort(key=lambda x: x[0])
    for param_count, num_layers, hidden_dims in arch_with_params:
        yield param_count, num_layers, hidden_dims


def main():
    # Dataset parameters
    nb_classes = 3
    nb_samples = 1875
    features_to_try = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]

    # Experiment parameters
    acc_threshold = 90
    repetitions = 5

    # Training parameters
    lr = 0.001
    bs = 64

    architectures = generate_architectures()

    for nb_features in features_to_try:
        print(f"\n ---> Experimenting using {nb_features} features")

        result = None
        for param_count, num_layers, hidden_dims in architecture_search_generator(
            nb_features, architectures, nb_classes
        ):
            print(
                f"Testing architecture: {num_layers} layers, {hidden_dims} hidden dims, {param_count} params"
            )

            mean_acc, std_acc, final_model = evaluate_architecture(
                nb_features, hidden_dims, nb_classes, nb_samples, repetitions, lr, bs
            )

            result = {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "num_layers": num_layers,
                "hidden_dims": hidden_dims,
                "param_count": param_count,
                "model": final_model,
            }

            if mean_acc >= acc_threshold:
                print(
                    f"\nThreshold achieved: {mean_acc:.2f} ± {std_acc:.2f} using {param_count} parameters"
                )
                break
            else:
                print(f"{mean_acc:.2f} < {acc_threshold}, moving to next architecture")

        # Prepare and save results
        experiment_dict = {
            "dataset": "spiral",
            "nb_features": nb_features,
            "cl best ACC": result["mean_acc"],
            "cl best ACC std": result["std_acc"],
            "nb layers": result["num_layers"],
            "hidden dims": result["hidden_dims"],
            "cl parameters": result["param_count"],
        }

        save_experiment_results(experiment_dict, f"clNN_SS_v4_bs{bs}_lr{lr}.json")


if __name__ == "__main__":
    main()
