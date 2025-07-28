"""
Training utilities and evaluation functions for VQC classification experiments.

This module provides comprehensive training, evaluation, and visualization functions
for comparing VQC performance with classical methods across different datasets.
Includes support for hyperparameter tuning, decision boundary visualization,
and performance comparison plots.
"""

import os
os.environ["WANDB_SILENT"] = "true"
from matplotlib.lines import Line2D
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from VQC import get_vqc, count_parameters
from data import get_linear, get_circle, get_moon, get_visual_sample, save_dataset_locally
from classical_models import get_mlp_deep, get_mlp_wide, count_svm_parameters
import wandb


def load_data():
    """
    Load pre-saved datasets from local files.
    
    Returns:
        tuple: (x_train, x_test, y_train, y_test) lists containing data for each dataset type
    """
    linear_data = torch.load("data/linear_dataset_vqc.pt")
    circular_data = torch.load("data/circular_dataset_vqc.pt")
    moon_data = torch.load("data/moon_dataset_vqc.pt")

    x_lin = linear_data["X_train"]
    x_lin_test = linear_data["X_test"]
    y_lin = linear_data["y_train"]
    y_lin_test = linear_data["y_test"]

    x_circ = circular_data["X_train"]
    x_circ_test = circular_data["X_test"]
    y_circ = circular_data["y_train"]
    y_circ_test = circular_data["y_test"]

    x_moon = moon_data["X_train"]
    x_moon_test = moon_data["X_test"]
    y_moon = moon_data["y_train"]
    y_moon_test = moon_data["y_test"]

    x_train = [x_lin, x_circ, x_moon]
    x_test = [x_lin_test, x_circ_test, x_moon_test]
    y_train = [y_lin, y_circ, y_moon]
    y_test = [y_lin_test, y_circ_test, y_moon_test]

    return x_train, x_test, y_train, y_test


def prepare_data(get_visu=False):
    """
    Generate and prepare datasets for training and testing.
    
    Args:
        get_visu (bool): Whether to generate and save data visualizations
        
    Returns:
        tuple: Training and test data for linear, circular, and moon datasets
    """
    # Labels have values 0 or 1
    x_lin, y_lin = get_linear(200, 2, class_sep=1.5)
    x_circ, y_circ = get_circle(200, noise=0.05)
    x_moon, y_moon = get_moon(200, noise=0.2)

    # For some reason, when we only take 100 samples points from get_linear, the distribution of the points changes drastically
    # so let's tak 200 data points and only keep half of them

    x_lin, _, y_lin, _ = train_test_split(x_lin, y_lin, test_size=0.5, random_state=42)
    x_circ, _, y_circ, _ = train_test_split(x_circ, y_circ, test_size=0.5, random_state=42)
    x_moon, _, y_moon, _ = train_test_split(x_moon, y_moon, test_size=0.5, random_state=42)

    x_lin_train, x_lin_test, y_lin_train, y_lin_test = train_test_split(x_lin, y_lin, test_size=0.4, random_state=42)
    
    # Convert data to PyTorch tensors
    x_lin_train = torch.FloatTensor(x_lin_train)
    y_lin_train = torch.FloatTensor(y_lin_train)
    x_lin_test = torch.FloatTensor(x_lin_test)
    y_lin_test = torch.FloatTensor(y_lin_test)

    scaler = StandardScaler()
    x_lin_train = torch.FloatTensor(scaler.fit_transform(x_lin_train))
    x_lin_test = torch.FloatTensor(scaler.transform(x_lin_test))

    print(f"Linear training set: {x_lin_train.shape[0]} samples, {x_lin_train.shape[1]} features")
    print(f"Linear test set: {x_lin_test.shape[0]} samples, {x_lin_test.shape[1]} features")
    
    x_circ_train, x_circ_test, y_circ_train, y_circ_test = train_test_split(x_circ, y_circ, test_size=0.4, random_state=42)

    # Convert data to PyTorch tensors
    x_circ_train = torch.FloatTensor(x_circ_train)
    y_circ_train = torch.FloatTensor(y_circ_train)
    x_circ_test = torch.FloatTensor(x_circ_test)
    y_circ_test = torch.FloatTensor(y_circ_test)

    scaler = StandardScaler()
    x_circ_train = torch.FloatTensor(scaler.fit_transform(x_circ_train))
    x_circ_test = torch.FloatTensor(scaler.transform(x_circ_test))

    print(f"Circular training set: {x_circ_train.shape[0]} samples, {x_circ_train.shape[1]} features")
    print(f"Circular test set: {x_circ_test.shape[0]} samples, {x_circ_test.shape[1]} features")
    
    x_moon_train, x_moon_test, y_moon_train, y_moon_test = train_test_split(x_moon, y_moon, test_size=0.4, random_state=42)

    # Convert data to PyTorch tensors
    x_moon_train = torch.FloatTensor(x_moon_train)
    y_moon_train = torch.FloatTensor(y_moon_train)
    x_moon_test = torch.FloatTensor(x_moon_test)
    y_moon_test = torch.FloatTensor(y_moon_test)

    scaler = StandardScaler()
    x_moon_train = torch.FloatTensor(scaler.fit_transform(x_moon_train))
    x_moon_test = torch.FloatTensor(scaler.transform(x_moon_test))

    print(f"Moon training set: {x_moon_train.shape[0]} samples, {x_moon_train.shape[1]} features")
    print(f"Moon test set: {x_moon_test.shape[0]} samples, {x_moon_test.shape[1]} features")

    if get_visu:
        get_visual_sample(torch.cat((x_lin_train, x_lin_test)), torch.cat((y_lin_train, y_lin_test)), "Data visualization of linear dataset")
        get_visual_sample(torch.cat((x_circ_train, x_circ_test)), torch.cat((y_circ_train, y_circ_test)), "Data visualization of circular dataset")
        get_visual_sample(torch.cat((x_moon_train, x_moon_test)), torch.cat((y_moon_train, y_moon_test)), "Data visualization of moon dataset")
        print("Visualizations saved locally")

    x_train = [x_lin_train, x_circ_train, x_moon_train]
    x_test = [x_lin_test, x_circ_test, x_moon_test]
    y_train = [y_lin_train, y_circ_train, y_moon_train]
    y_test = [y_lin_test, y_circ_test, y_moon_test]

    save_dataset_locally("data/linear_dataset_vqc.pt", x_lin_train, x_lin_test, y_lin_train, y_lin_test)
    save_dataset_locally("data/circular_dataset_vqc.pt", x_circ_train, x_circ_test, y_circ_train, y_circ_test)
    save_dataset_locally("data/moon_dataset_vqc.pt", x_moon_train, x_moon_test, y_moon_train, y_moon_test)

    return x_train, x_test, y_train, y_test


def train_model(model, X_train, y_train, X_test, y_test, model_name, args, extra_visu=False):
    """Train a model and return training metrics"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)
    criterion = torch.nn.MSELoss()  # loss from paper

    if args.log_wandb:
        # Wandb start
        unique_id = wandb.util.generate_id()
        # Initialize a new W&B run
        wandb.init(
            project="VQC gan paper",
            name = f"{args.model_type}_{args.dataset_name}_{unique_id}",
            group=f"{args.model_type}_{args.dataset_name}",
            tags=[f"{args.model_type}", f"{args.dataset_name}"],
            config={
                "dataset": args.dataset_name,
                "initial_state": args.initial_state,
                "learning_rate": args.lr,
                "epochs": args.n_epochs,
                "batch_size": args.batch_size,
                "activation": args.activation,
                "no_bunching": args.no_bunching,
                "alpha": args.alpha,
                "betas": args.betas,
                "circuit": args.circuit,
                "scale_type": args.scale_type,
                "regu_on": args.regu_on
            },
        )

    losses = []
    train_accuracies = []
    test_accuracies = []

    model.train()

    pbar = tqdm(range(args.n_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0

        for i in range(0, X_train.size()[0], args.batch_size):
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            # If softmax activation, we have to format the labels in 2-dimensions
            if args.activation == "softmax":
                # One-hot encode
                batch_y = F.one_hot(batch_y.squeeze().long(), num_classes=2).float()  # shape : (batch_size, 2)

            outputs = model(batch_x)

            # Get the right parameters for regularization
            if args.regu_on == "linear":
                # Get the observable parameters
                if args.activation == "none":
                    obs = list(model[-1].parameters())
                else:
                    obs = list(model[-2].parameters())
                # Flatten all tensors and concatenate into a single vector
                obs_vector = torch.cat([p.view(-1) for p in obs])
            elif args.regu_on == "all":
                # Get the observable parameters
                obs = list(model.parameters())
                # Flatten all tensors and concatenate into a single vector
                obs_vector = torch.cat([p.view(-1) for p in obs])
            else:
                raise NotImplementedError

            loss = criterion(outputs.squeeze(), batch_y.squeeze()) + args.alpha * torch.linalg.vector_norm(obs_vector)**2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (X_train.size()[0] // args.batch_size)
        losses.append(avg_loss)

        # Evaluation
        model.eval()
        with torch.no_grad():
            train_outputs = model(X_train)

            if args.activation == "softmax":
                train_preds = torch.argmax(train_outputs, dim=1)
            else:
                train_preds = torch.round(train_outputs)

            train_acc = accuracy_score(y_train.numpy(), train_preds.cpu().numpy())
            train_accuracies.append(train_acc)

            test_outputs = model(X_test)

            if args.activation == "softmax":
                test_preds = torch.argmax(test_outputs, dim=1)
            else:
                test_preds = torch.round(test_outputs)

            test_acc = accuracy_score(y_test.numpy(), test_preds)
            test_accuracies.append(test_acc)

            pbar.set_description(f"Training {model_name} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        if args.log_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_acc": train_acc,
                "test_acc": test_acc,
            })

        model.train()

    if args.log_wandb:
        # Wandb end
        wandb.finish()

    if extra_visu:
        visualize_decision_boundary(model, X_train, y_train, X_test, y_test, args)
    return {
        'losses': losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'final_test_acc': test_accuracies[-1]
    }


def train_model_multiple_runs(type, args):
    """Train model multiple times and return results"""

    try:
        X_trains, X_tests, y_trains, y_tests = load_data()
        print("Data loaded")
    except FileNotFoundError:
        print("Datasets not found, generating data and saving it locally at ./data/...\n")
        X_trains, X_tests, y_trains, y_tests = prepare_data(get_visu=True)
    datasets = ["linear", "circular", "moon"]
    results = {}

    for i, dataset in enumerate(datasets):
        X_train = X_trains[i]
        X_test = X_tests[i]
        y_train = y_trains[i]
        y_test = y_tests[i]

        if type == "vqc":
            print(f"\nTraining VQC with dataset {dataset} ({args.num_runs} runs):")
            models = []
            model_runs = []

            for run in range(args.num_runs):
                visualize_circuit = (run == 0)  # visualize circuit only for the first run

                # Create a fresh instance of the model for each run
                model = get_vqc(args.m, args.input_size, args.initial_state, activation=args.activation,
                                no_bunching=args.no_bunching, circuit=args.circuit, visualize=visualize_circuit,
                                scale_type=args.scale_type)

                num_params = count_parameters(model)

                print(f"  Run {run + 1}/{args.num_runs}, VQC has {num_params} parameters...")
                args.set_dataset_name(dataset)
                run_results = train_model(model, X_train, y_train, X_test, y_test, f"VQC-run{run + 1}", args)
                models.append(model)
                model_runs.append(run_results)

        elif type[:3] == "mlp":
            if type == "mlp_wide":
                network_type = "wide"
                get_mlp = get_mlp_wide
            elif type == "mlp_deep":
                network_type = "deep"
                get_mlp = get_mlp_deep
            else:
                raise NotImplementedError
            print(f"\nTraining MLP ({network_type}) with dataset {dataset} ({args.num_runs} runs):")
            models = []
            model_runs = []

            for run in range(args.num_runs):
                # Create a fresh instance of the model for each run
                model = get_mlp(args.input_size, activation=args.activation)

                num_params = count_parameters(model)

                print(f"  Run {run + 1}/{args.num_runs}, MLP has {num_params} parameters...")
                args.set_dataset_name(dataset)
                run_results = train_model(model, X_train, y_train, X_test, y_test, f"MLP-run{run + 1}", args)
                models.append(model)
                model_runs.append(run_results)

        else:
            if type == "svm_lin":
                kernel = "linear"
            elif type == "svm_rbf":
                kernel = "rbf"
            else:
                raise NotImplementedError(f"Unknown type of model: {type}")

            print(f"\nTraining SVM ({kernel}) with dataset {dataset} ({args.num_runs} runs):")
            models = []
            model_runs = []

            for run in range(args.num_runs):
                # Create a fresh instance of the model for each run
                model = SVC(kernel=kernel, gamma="scale")
                model.fit(X_train, y_train)

                num_params = count_svm_parameters(model, kernel)
                print(f"  Run {run + 1}/{args.num_runs}, SVM has {num_params} parameters...")
                args.set_dataset_name(dataset)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                models.append(model)
                run_results = {
                    'losses': [0.0] * args.n_epochs,
                    'train_accuracies': [0.0] * args.n_epochs,
                    'test_accuracies': [0.0] * args.n_epochs,
                    'final_test_acc': accuracy
                }
                model_runs.append(run_results)


        best_run = torch.argmax(torch.Tensor([run["final_test_acc"] for run in model_runs]))
        best_model = models[best_run]
        best_acc = torch.max(torch.Tensor([run["final_test_acc"] for run in model_runs]))
        visualize_decision_boundary(type, best_model, X_train, y_train, X_test, y_test, best_acc, args)

        # Store all runs for this model
        results[dataset] = {
            'runs': model_runs,
            # Calculate aggregate statistics
            'avg_final_test_acc': sum(run['final_test_acc'] for run in model_runs) / args.num_runs
        }

    for dataset_name, model_data in results.items():
        # Calculate statistics across runs
        final_accs = [run['final_test_acc'] for run in model_data['runs']]
        avg_acc = sum(final_accs) / len(final_accs)
        std_acc = (sum((acc - avg_acc) ** 2 for acc in final_accs) / len(final_accs)) ** 0.5

        if args.log_wandb:
            unique_id = wandb.util.generate_id()
            wandb.init(
                project="VQC gan paper",
                name=f"{args.model_type}_{dataset_name}_final_{unique_id}",
                group=f"{args.model_type}_{dataset_name}_final",
                tags=[f"{args.model_type}", f"{dataset_name}", "final"],
                config={
                    "dataset": dataset_name,
                    "initial_state": args.initial_state,
                    "learning_rate": args.lr,
                    "epochs": args.n_epochs,
                    "batch_size": args.batch_size,
                    "activation": args.activation,
                    "no_bunching": args.no_bunching,
                    "alpha": args.alpha,
                    "betas": args.betas,
                    "circuit": args.circuit,
                    "scale_type": args.scale_type,
                    "regu_on": args.regu_on
                },
            )
            wandb.log({"Mean final test accuracy:": avg_acc, "Final test accuracy std": std_acc})
            wandb.finish()

    return results


def train_vqc_multiple_runs(args):
    args.set_model_type(f"vqc_{args.initial_state}")
    return train_model_multiple_runs("vqc", args)


def train_mlp_multiple_runs(args, network_type="wide"):
    if network_type == "wide":
        args.set_model_type("mlp_wide")
        return train_model_multiple_runs("mlp_wide", args)
    elif network_type == "deep":
        args.set_model_type("mlp_deep")
        return train_model_multiple_runs("mlp_deep", args)
    else:
        raise NotImplementedError


def train_svm_multiple_runs(args, kernel_type="lin"):
    if kernel_type == "lin":
        args.set_model_type("svm_lin")
        return train_model_multiple_runs("svm_lin", args)
    elif kernel_type == "rbf":
        args.set_model_type("svm_rbf")
        return train_model_multiple_runs("svm_rbf", args)
    else:
        raise NotImplementedError


def visualize_decision_boundary(type, model, X_train, y_train, X_test, y_test, acc, args, resolution=100):
    # Convert to numpy if X_train is a tensor
    if isinstance(X_train, torch.Tensor):
        X_train_np = X_train.cpu().numpy()
    else:
        X_train_np = X_test

    # Convert to numpy if X_test is a tensor
    if isinstance(X_test, torch.Tensor):
        X_test_np = X_test.cpu().numpy()
    else:
        X_test_np = X_test

    # Determine plot boundaries
    x_min1, x_max1 = X_train_np[:, 0].min() - 1, X_train_np[:, 0].max() + 1
    y_min1, y_max1 = X_train_np[:, 1].min() - 1, X_train_np[:, 1].max() + 1
    x_min2, x_max2 = X_test_np[:, 0].min() - 1, X_test_np[:, 0].max() + 1
    y_min2, y_max2 = X_test_np[:, 1].min() - 1, X_test_np[:, 1].max() + 1

    x_min = min(x_min1, x_min2)
    x_max = max(x_max1, x_max2)
    y_min = min(y_min1, y_min2)
    y_max = max(y_max1, y_max2)

    # Create a meshgrid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    if type == "vqc" or type == "mlp_wide" or type == "mlp_deep":
        # Ensure model is in eval mode and on the correct device
        model.eval()

        # Convert to torch tensor
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

        # Predict with the model (no gradient needed)
        with torch.no_grad():
            preds = model(grid_tensor).squeeze()

        if args.activation == "none":
            probs = preds.cpu().numpy()
        elif args.activation == "softmax":
            probs = preds[:, 1].cpu().numpy()
        else:
            probs = preds.cpu().numpy()

    elif type == "svm_lin" or type == "svm_rbf":
        probs = model.predict(grid_points)

    else:
        raise NotImplementedError

    # Convert to 0 or 1 prediction
    pred_labels = (probs > 0.5).astype(int)

    # Plot decision boundary
    plt.contourf(xx, yy, pred_labels.reshape(xx.shape), alpha=0.3, cmap=plt.cm.RdBu)

    # Plot train points
    plt.scatter(
        X_train_np[:, 0], X_train_np[:, 1],
        c=y_train, cmap=plt.cm.RdBu, s=30, marker="o", label="Training data"
    )

    # Plot test points
    plt.scatter(
        X_test_np[:, 0], X_test_np[:, 1],
        c=y_test, cmap=plt.cm.RdBu, s=30, marker="x", label="Test data"
    )

    formatted_acc = f"{acc:.2f}"
    plt.text(0.01, 0.99, formatted_acc,
             transform=plt.gca().transAxes,  # get current Axes and use its transform
             ha='left', va='top', fontsize=16)

    plt.xlabel("x1", size=16)
    plt.ylabel("x2", size=16)
    plt.title(f"Decision Boundary of {args.model_type}\non {args.dataset_name} dataset", size=16)
    # plt.legend(*scatter.legend_elements(), title="Class")
    # plt.legend()
    legend_elements = [
        Patch(facecolor='blue', label='Label 0'),
        Patch(facecolor='red', label='Label 1'),
        Line2D([0], [0], marker='o', color='gray', label='Training data', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='x', color='gray', label='Test data', markerfacecolor='gray', markersize=8)
    ]

    plt.legend(handles=legend_elements, loc='upper right')
    plt.savefig(f"./results/decision_boundary_{args.model_type}_{args.dataset_name}.png")
    plt.clf()
    return


def visualize_results(results):
    """Plot training curves with average and envelope for each dataset"""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    datasets = ["linear", "circular", "moon"]
    datasets_maj = ["Linear", "Circular", "Moon"]
    colors = ["red", "green", "blue"]

    # Plot each metric
    for i, dataset in enumerate(datasets):
        color = colors[i]
        linestyle = '-'

        # Get data from all runs
        losses_runs = [run['losses'] for run in results[dataset]['runs']]
        train_acc_runs = [run['train_accuracies'] for run in results[dataset]['runs']]
        test_acc_runs = [run['test_accuracies'] for run in results[dataset]['runs']]

        # Calculate mean values across all runs
        epochs = len(losses_runs[0])
        mean_losses = [sum(run[i] for run in losses_runs) / len(losses_runs) for i in range(epochs)]
        mean_train_acc = [sum(run[i] for run in train_acc_runs) / len(train_acc_runs) for i in range(epochs)]
        mean_test_acc = [sum(run[i] for run in test_acc_runs) / len(test_acc_runs) for i in range(epochs)]

        # Calculate min and max values for the envelope
        min_losses = [min(run[i] for run in losses_runs) for i in range(epochs)]
        max_losses = [max(run[i] for run in losses_runs) for i in range(epochs)]

        min_train_acc = [min(run[i] for run in train_acc_runs) for i in range(epochs)]
        max_train_acc = [max(run[i] for run in train_acc_runs) for i in range(epochs)]

        min_test_acc = [min(run[i] for run in test_acc_runs) for i in range(epochs)]
        max_test_acc = [max(run[i] for run in test_acc_runs) for i in range(epochs)]

        # Plot mean lines
        ax1.plot(mean_losses, label=datasets_maj[i], color=color, linestyle=linestyle, linewidth=2)
        ax2.plot(mean_train_acc, label=datasets_maj[i], color=color, linestyle=linestyle, linewidth=2)
        ax3.plot(mean_test_acc, label=datasets_maj[i], color=color, linestyle=linestyle, linewidth=2)

        # Plot envelopes (filled area between min and max)
        epochs_range = range(epochs)
        ax1.fill_between(epochs_range, min_losses, max_losses, color=color, alpha=0.2)
        ax2.fill_between(epochs_range, min_train_acc, max_train_acc, color=color, alpha=0.2)
        ax3.fill_between(epochs_range, min_test_acc, max_test_acc, color=color, alpha=0.2)

    # Customize plots
    for ax, title in zip([ax1, ax2, ax3], ['Training Loss', 'Training Accuracy', 'Test Accuracy']):
        ax.set_title(title + " for a VQC\non different datasets", fontsize=14, pad=10)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(title.split()[-1], fontsize=12)
        ax.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig("./results/vqc_results.png")
    plt.clf()
    return


def write_summary_statistics(all_results, args):
    """Print and save locally hyperparameters info and summary statistics for all datasets"""
    s = "----- Hyperparameters information -----\n"
    s += f"m = {args.m}\ninput_size = {args.input_size}\ninitial_state = {args.initial_state}\n\n"
    s += (f"activation = {args.activation}\nno_bunching = {args.no_bunching}\nnum_runs = {args.num_runs}\nn_epochs = "
          f"{args.n_epochs}\nbatch_size = {args.batch_size}\nlr = {args.lr}\nalpha = {args.alpha}\nbetas = {args.betas}"
          f"\ncircuit = {args.circuit}\nscale_type = {args.scale_type}\nregu_on = {args.regu_on}\n")
    s += "\n----- Model Comparison Results -----\n"

    for dataset_name, model_data in all_results.items():
        # Calculate statistics across runs
        final_accs = [run['final_test_acc'] for run in model_data['runs']]
        avg_acc = sum(final_accs) / len(final_accs)
        min_acc = min(final_accs)
        max_acc = max(final_accs)
        std_acc = (sum((acc - avg_acc) ** 2 for acc in final_accs) / len(final_accs)) ** 0.5

        s += f"VQC with dataset {dataset_name}:\n"
        s += f"  Final Test Accuracy: {avg_acc:.4f} ± {std_acc:.4f} (min: {min_acc:.4f}, max: {max_acc:.4f})\n\n"

    print(s)
    with open("results/vqc_info_and_stats.txt", "w") as f:
        f.write(s)
    return
