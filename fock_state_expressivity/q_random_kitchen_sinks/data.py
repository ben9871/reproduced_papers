import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_moon_dataset(args):
    """
    Return moon dataset x and y.
    x : [n_samples, 2]
    y : [n_samples, ] of value 0 or 1
    """
    x, y = make_moons(
        n_samples=args.n_samples, noise=args.noise, random_state=args.random_state
    )
    return np.array(x), np.array(y)


def scale_dataset(x_train, x_test, scaling):
    if scaling == "Standard":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")
    return x_train, x_test


def split_train_test(x, y, test_prop, random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_prop, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


def get_target_function(x_r_i_s):
    return np.sqrt(2) * np.cos(x_r_i_s)


def visualize_dataset(x_train, x_test, y_train, y_test):
    plt.figure(figsize=(6, 6))

    # Plot training data (circle marker 'o')
    plt.scatter(
        x_train[y_train == 0][:, 0],
        x_train[y_train == 0][:, 1],
        color="red",
        marker="o",
        label="Class 0 - Train",
        s=80,
    )

    # Plot test data (cross marker 'x')
    plt.scatter(
        x_test[y_test == 0][:, 0],
        x_test[y_test == 0][:, 1],
        color="red",
        marker="x",
        label="Class 0 - Test",
        s=80,
    )

    plt.scatter(
        x_train[y_train == 1][:, 0],
        x_train[y_train == 1][:, 1],
        color="blue",
        marker="o",
        label="Class 1 - Train",
        s=80,
    )

    plt.scatter(
        x_test[y_test == 1][:, 0],
        x_test[y_test == 1][:, 1],
        color="blue",
        marker="x",
        label="Class 1 - Test",
        s=80,
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Moons Dataset: Train and Test Split")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data/moons_dataset.png")
    plt.close()
    return


def save_dataset(x_train, x_test, y_train, y_test):
    np.save("./data/moons_x_train.npy", x_train)
    np.save("./data/moons_x_test.npy", x_test)
    np.save("./data/moons_y_train.npy", y_train)
    np.save("./data/moons_y_test.npy", y_test)
    return


def visualize_kernel(k, y_train, args, q_approx=True):
    # Sort by class for visual clarity (optional but very useful)
    sorted_indices = np.argsort(y_train)
    k_sorted = k[sorted_indices][:, sorted_indices]
    y_sorted = y_train[sorted_indices]

    # Determine the class boundary (index where class 1 starts)
    split_index = np.sum(y_sorted == y_sorted[0])  # assuming y_sorted[0] == 0

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(k_sorted, cmap="viridis", cbar=True)

    # Add horizontal and vertical black lines
    ax.axhline(split_index, color="black", lw=1.5)
    ax.axvline(split_index, color="black", lw=1.5)

    if q_approx:
        plt.title(
            f"Q-Kernel Matrix (sorted by class) with R = {args.r}, $\\gamma$ = {args.gamma}"
        )
    else:
        plt.title(
            f"Kernel Matrix (sorted by class) with R = {args.r}, $\\gamma$ = {args.gamma}"
        )
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    if q_approx:
        plt.savefig(f"./results/q_kernel_matrix_visu_r_{args.r}_gamma_{args.gamma}.png")
    else:
        plt.savefig(f"./results/kernel_matrix_visu_r_{args.r}_gamma_{args.gamma}.png")
    wandb.log({"Kernel Matrix": wandb.Image(plt.gcf())})
    plt.close()
    return
