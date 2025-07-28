import torch
import torch.nn as nn
import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from data import prepare_data, get_visual_sample
from model import create_quantum_layer
import matplotlib.pyplot as plt
import wandb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
from matplotlib.patches import Patch


def train_model(model, X_train, y_train, model_name, args):
    """Train a model and return training metrics"""
    if args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.MSELoss()

    losses = []
    train_mses = []

    model.train()

    pbar = tqdm.tqdm(range(args.num_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        permutation = torch.randperm(X_train.size()[0])
        total_loss = 0

        for i in range(0, X_train.size()[0], args.batch_size):
            if args.shuffle_train:
                indices = permutation[i:i + args.batch_size]
                batch_x, batch_y = X_train[indices], y_train[indices]
            else:
                start_index = i
                end_index = i + args.batch_size
                batch_x = X_train[start_index:end_index]
                batch_y = y_train[start_index:end_index]

            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())

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
            train_mse = mean_squared_error(y_train.numpy(), train_outputs)
            train_mses.append(train_mse)

            wandb.log({"train_mse": train_mse, "epoch": epoch})

            pbar.set_description(f"Training {model_name} - Loss: {avg_loss:.4f}, Train MSE: {train_mse:.4f}")

        model.train()

    return {
        'losses': losses,
        'train_mses': train_mses,
    }

def train_models_multiple_runs(num_photons, colors, X_train, ys_info, args):
    """Train all models multiple times and return results"""
    all_results = {}
    ys = ys_info["ys"]
    names = ys_info["names"]

    assert names[0] == "std = 1.00"

    for y, name in zip(ys, names):
        results = {}
        models = []
        for n, color in zip(num_photons, colors):
            print(f"\nTraining Q-Gaussian kernel with {n} photons ({args.num_runs} runs) for Gaussian with {name}:")
            pending_models = []
            model_runs = []

            for run in range(args.num_runs):
                unique_id = wandb.util.generate_id()
                wandb.init(
                    project="Q-Gaussian kernels - Gan paper",
                    group=f"{name}, num_photons={n}",  # optional: groups runs by std dev
                    name=f"{name}_n{n}_r{run}_{unique_id}",  # optional: clear naming
                    config={
                        "std": float(name[-4:]),
                        "n_photons": n,
                        "seed": run,
                        "num_runs": args.num_runs,
                        "num_epochs": args.num_epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "betas": args.betas,
                        "weight_decay": args.weight_decay,
                        "train_circuit": args.train_circuit,
                        "scale_type": args.scale_type,
                        "circuit": args.circuit,
                        "no_bunching": args.no_bunching,
                        "optimizer": args.optimizer,
                        "shuffle_train": args.shuffle_train,
                    }
                )

                # Create a fresh instance of the model for each run
                mzi = create_quantum_layer(n, args)

                print(f"  Run {run+1}/{args.num_runs}...")
                run_results = train_model(mzi, torch.tensor(X_train, dtype=torch.float).unsqueeze(-1), torch.tensor(y, dtype=torch.float), f"MZI_{n}-run{run+1}", args)
                pending_models.append(mzi)
                model_runs.append(run_results)

                wandb.log({"final_train_mse": run_results["train_mses"][-1]})
                wandb.finish()

            # Find and keep the best model for each number of photons
            index = torch.argmin(torch.tensor([model_run["train_mses"][-1] for model_run in model_runs]))

            models.append(pending_models[index])
            # Store all runs for this model
            results[f"MZI_{n}"] = {
                "runs": model_runs,
                "color": color,
            }
        all_results[f"{name}"] = {
            "results": results,
            "models": models,
        }

    return all_results, unique_id

def visualize_learned_function(results, num_photons, x_on_pi, delta, ys_info, args, unique_id):
    """Visualize learned function of different models to compare them with the target function, a Gaussian"""
    wandb.init(
        project="Q-Gaussian kernels - Gan paper",
        group="Overview",
        name=f"{unique_id}",
        config={
            "num_runs": args.num_runs,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "betas": args.betas,
            "weight_decay": args.weight_decay,
            "train_circuit": args.train_circuit,
            "scale_type": args.scale_type,
            "circuit": args.circuit,
            "no_bunching": args.no_bunching,
            "optimizer": args.optimizer,
            "shuffle_train": args.shuffle_train,
        }
    )

    mses = []

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    i = 0
    j = 0
    ys = ys_info["ys"]
    y_names = ys_info["names"]
    sigmas = [1.00, 0.50, 0.33, 0.25]
    circuit_names = [f"MZI_{n}" for n in num_photons]

    for y, y_name, sigma in zip(ys, y_names, sigmas):
        axis = axs[i // 2][j % 2]
        y_results = results[f"{y_name}"]["results"]
        y_models = results[f"{y_name}"]["models"]
        for circuit_name, model in zip(circuit_names, y_models):
            model_results = y_results[circuit_name]
            runs_results = model_results["runs"]
            for run_results in runs_results:
                mses.append(run_results["train_mses"][-1])
            color = model_results["color"]
            model.eval()
            with torch.no_grad():
                output = model(torch.tensor(delta, dtype=torch.float).unsqueeze(-1))
            axis.plot(x_on_pi, output.detach().numpy(), label=f"n = {circuit_name[4:]}", color=color, linewidth=1)

        axis.scatter(x_on_pi, y, s=10, color="k")

        axis.set_xlabel('x / pi')
        axis.set_ylabel('y')
        axis.grid(True)
        if i == 0 and j == 0:
            axis.legend(loc="upper right")
        axis.title.set_text(fr"$\sigma$ = {sigma}")
        i += 1
        j += 1
    fig.suptitle("Approximating Gaussian kernels of different standard deviations\nwith 2 mode circuits of varying number of photons (n)", fontsize=14)
    # To save the figure locally
    #plt.savefig(f"./learned_q_gaussian_kernels.png")

    wandb.log({"learned_functions": wandb.Image(fig)})
    assert len(mses) == len(circuit_names) * len(sigmas) * args.num_runs
    wandb.log({"Mean final MSE": torch.mean(torch.tensor(mses, dtype=torch.float)).item()})
    wandb.finish()
    #plt.show()
    return

def save_models(results, num_photons, ys_info):
    ys = ys_info["ys"]
    y_names = ys_info["names"]
    sigmas = [1.00, 0.50, 0.33, 0.25]
    circuit_names = [f"MZI_{n}" for n in num_photons]

    for y, y_name, sigma in zip(ys, y_names, sigmas):
        y_models = results[f"{y_name}"]["models"]
        for circuit_name, model in zip(circuit_names, y_models):
            torch.save(model.state_dict(), f'./models/n{circuit_name[4:]}_std{sigma}.pth')

    return

# For classification

def calculate_delta(x1, x2):
    """
    Computes squared Euclidean distances between each pair of vectors in x1 and x2.
    x1: Tensor of shape (n1, d)
    x2: Tensor of shape (n2, d)
    Returns: Tensor of shape (n1, n2) with delta[i, j] = ||x1[i] - x2[j]||^2
    """
    # Ensure 2D input
    assert x1.ndim == 2 and x2.ndim == 2, "Inputs must be 2D tensors"

    # Use broadcasting to compute pairwise squared Euclidean distances
    diff = x1[:, None, :] - x2[None, :, :]     # shape (n1, n2, d)
    delta = torch.sum(diff ** 2, dim=2)        # shape (n1, n2)

    # Optional sanity checks
    assert delta.shape[0] == x1.size(0), "First dimension of delta is off"
    assert delta.shape[1] == x2.size(0), "Second dimension of delta is off"

    return delta

def get_kernel(model, delta):
    """
    Efficiently apply `model` to each element in the `delta` matrix.
    Assumes model maps a scalar input to a scalar output.
    """
    model.eval()
    with torch.no_grad():
        flat_input = delta.view(-1, 1)  # shape (n1 * n2, 1)
        output = model(flat_input)     # shape (n1 * n2, 1) or (n1 * n2,)
        kernel_matrix = output.view(delta.shape)
    return kernel_matrix

def train_svm(k_train, k_test, y_train, y_test, kernel='precomputed', gamma=0.5):
    if kernel == 'precomputed':
        clf = SVC(kernel='precomputed')
    elif kernel == 'rbf':
        clf = SVC(kernel='rbf', gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel type: {kernel}")

    clf.fit(k_train.numpy(), y_train.numpy())
    y_pred = clf.predict(k_test.numpy())
    accuracy = accuracy_score(y_test.numpy(), y_pred)
    return accuracy

def train_classif(args, scaling_factor, ys_info, num_photons, type='quantum'):
    initial_time = time.time()
    x_train, x_test, y_train, y_test = prepare_data(scaling_factor=scaling_factor)
    # Visualize training data
    get_visual_sample(x_train, y_train, x_test, y_test, title=["Circular dataset", "Moon dataset", "Blob dataset"])
    accs = {"circular_acc": [], "moon_acc": [], "blob_acc": [], "std_name": [], "n_photons": []}
    std_names = ys_info["names"]

    if type == 'quantum':
        for std_name in std_names:
            for i in range(5):
                n_photons = (i + 1) * 2
                model = create_quantum_layer(num_photons=n_photons, args=args)
                std = std_name[6:]
                model.load_state_dict(torch.load(f"./models/n{n_photons}_std{std}.pth"))

                # Calculate delta
                delta_train_circ = calculate_delta(x_train[0], x_train[0])
                delta_test_circ = calculate_delta(x_test[0], x_train[0])
                delta_train_moon = calculate_delta(x_train[1], x_train[1])
                delta_test_moon = calculate_delta(x_test[1], x_train[1])
                delta_train_blob = calculate_delta(x_train[2], x_train[2])
                delta_test_blob = calculate_delta(x_test[2], x_train[2])

                # Calculate kernel
                kernel_train_circ = get_kernel(model, delta_train_circ)
                kernel_test_circ = get_kernel(model, delta_test_circ)
                kernel_train_moon = get_kernel(model, delta_train_moon)
                kernel_test_moon = get_kernel(model, delta_test_moon)
                kernel_train_blob = get_kernel(model, delta_train_blob)
                kernel_test_blob = get_kernel(model, delta_test_blob)

                # Train SVM
                circular_acc = train_svm(kernel_train_circ, kernel_test_circ, y_train[0], y_test[0], kernel='precomputed')
                moon_acc = train_svm(kernel_train_moon, kernel_test_moon, y_train[1], y_test[1], kernel='precomputed')
                blob_acc = train_svm(kernel_train_blob, kernel_test_blob, y_train[2], y_test[2], kernel='precomputed')

                accs["circular_acc"].append(circular_acc)
                accs["moon_acc"].append(moon_acc)
                accs["blob_acc"].append(blob_acc)
                accs["std_name"].append(std_name)
                accs["n_photons"].append(n_photons)

    elif type == 'rbf':
        for std_name in std_names:
            std = float(std_name[-4:])
            gamma = 1.0 / (2 * std**2)
            circular_acc = train_svm(x_train[0], x_test[0], y_train[0], y_test[0], kernel='rbf', gamma=gamma)
            moon_acc = train_svm(x_train[1], x_test[1], y_train[1], y_test[1], kernel='rbf', gamma=gamma)
            blob_acc = train_svm(x_train[2], x_test[2], y_train[2], y_test[2], kernel='rbf', gamma=gamma)

            accs['circular_acc'].append(circular_acc)
            accs['moon_acc'].append(moon_acc)
            accs['blob_acc'].append(blob_acc)
            accs['std_name'].append(std_name)
            accs['n_photons'].append(-1)

    else:
        raise ValueError(f"Unknown kernel type: {type}")

    final_time = time.time()
    return accs, final_time - initial_time

def visualize_accuracies(q_results, classical_results):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Map each std to a subplot index
    std_to_pos = {
        1.0: (0, 0),
        0.5: (0, 1),
        0.33: (1, 0),
        0.25: (1, 1)
    }

    datasets = ["circular", "moon", "blob"]
    quantum_colors = ["red", "blue", "green"]
    bar_width = 0.25

    unique_stds = sorted(set(q_results["std_name"]), reverse=True)  # just in case

    for std in unique_stds:
        std_str = std
        std = float(std[6:])
        if std not in std_to_pos:
            print(f"std skipped: {std}")
            continue  # skip stds not in layout
        axis = axs[std_to_pos[std]]

        # Get quantum data for this std
        std_indices = [j for j, s in enumerate(q_results["std_name"]) if s == std_str]
        n_photon_vals = sorted(set([q_results["n_photons"][j] for j in std_indices]))

        num_groups = len(n_photon_vals) + 1  # +1 for classical comparison
        positions = np.arange(num_groups)

        for k, dataset in enumerate(datasets):
            y_vals = []
            for n in n_photon_vals:
                idx = next(j for j in std_indices if q_results["n_photons"][j] == n)
                y = q_results[f"{dataset}_acc"][idx]
                y_vals.append(y)

            classical_idx = classical_results["std_name"].index(std_str)
            classical_y = classical_results[f"{dataset}_acc"][classical_idx]
            y_vals.append(classical_y)

            x_pos = positions + k * bar_width
            for j, y in enumerate(y_vals):
                if j < len(n_photon_vals):
                    # Quantum bar
                    axis.bar(
                        x_pos[j], y,
                        width=bar_width,
                        color=quantum_colors[k]
                    )
                else:
                    # Classical bar
                    axis.bar(
                        x_pos[j], y,
                        width=bar_width,
                        color=quantum_colors[k],
                    )
                # Add text above each bar
                axis.text(
                    x_pos[j], y + 0.01, f"{y:.2f}",
                    ha='center', va='bottom', fontsize=6
                )

        x_labels = [str(n) + " photons" for n in n_photon_vals] + ["Classical"]
        tick_positions = positions + bar_width
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(x_labels, rotation=45)
        axis.set_title(f"σ = {std}")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1)

        # Draw vertical line between quantum and classical bars
        classical_start = positions[-1] - bar_width / 2  # Left edge of classical bars
        axis.axvline(
            x=classical_start - 0.12,  # small offset for aesthetics
            color='black',
            linestyle='--',
            linewidth=1
        )

    plt.suptitle(
        "SVM classification accuracy across datasets\ncomparing approximated and exact Gaussian kernels",
        fontsize=14
    )
    handles, labels = axs[0, 0].get_legend_handles_labels()
    legend_patches = [
        Patch(facecolor=quantum_colors[i], label=datasets[i].capitalize())
        for i in range(len(datasets))
    ]

    fig.legend(
        handles=legend_patches,
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        title="Dataset"
    )
    #fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), title="Dataset")
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig("./results/svm_accuracies_quantum_kernel.png", bbox_inches='tight', dpi=600)
    '''plt.show()
    plt.clf()'''
    return
