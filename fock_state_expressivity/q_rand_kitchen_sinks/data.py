from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb

def get_moon_dataset(args):
    """
    Return moon dataset X and y.
    X : [n_samples, 2]
    y : [n_samples, ] of value 0 or 1
    """
    X, y = make_moons(n_samples=args.n_samples, noise=args.noise, random_state=args.random_state)
    return np.array(X), np.array(y)

def scale_dataset(X_train, X_test, scaling):
    if scaling == 'Standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaling == 'MinMax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        raise ValueError(f'Unknown scaling method: {scaling}')
    return X_train, X_test

def split_train_test(X, y, test_prop, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_prop, random_state=random_state)
    return X_train, X_test, y_train, y_test

def get_target_function(x_r_i_s):
    return np.sqrt(2) * np.cos(x_r_i_s)


def visualize_dataset(X_train, X_test, y_train, y_test):
    plt.figure(figsize=(6, 6))

    # Plot training data (circle marker 'o')
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
                color='red', marker='o', label='Class 0 - Train', s=80)

    # Plot test data (cross marker 'x')
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1],
                color='red', marker='x', label='Class 0 - Test', s=80)

    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                color='blue', marker='o', label='Class 1 - Train', s=80)

    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1],
                color='blue', marker='x', label='Class 1 - Test', s=80)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Moons Dataset: Train and Test Split')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./data/moons_dataset.png')
    plt.close()
    return

def save_dataset(X_train, X_test, y_train, y_test):
    np.save('./data/moons_X_train.npy', X_train)
    np.save('./data/moons_X_test.npy', X_test)
    np.save('./data/moons_y_train.npy', y_train)
    np.save('./data/moons_y_test.npy', y_test)
    return

def visualize_kernel(K, y_train, args, q_approx=True):
    # Sort by class for visual clarity (optional but very useful)
    sorted_indices = np.argsort(y_train)
    K_sorted = K[sorted_indices][:, sorted_indices]
    y_sorted = y_train[sorted_indices]

    # Determine the class boundary (index where class 1 starts)
    split_index = np.sum(y_sorted == y_sorted[0])  # assuming y_sorted[0] == 0

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(K_sorted, cmap='viridis', cbar=True)

    # Add horizontal and vertical black lines
    ax.axhline(split_index, color='black', lw=1.5)
    ax.axvline(split_index, color='black', lw=1.5)

    if q_approx:
        plt.title(f"Q-Kernel Matrix (sorted by class) with R = {args.r}, $\\gamma$ = {args.gamma}")
    else:
        plt.title(f"Kernel Matrix (sorted by class) with R = {args.r}, $\\gamma$ = {args.gamma}")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Index")
    if q_approx:
        plt.savefig(f'./results/q_kernel_matrix_visu_r_{args.r}_gamma_{args.gamma}.png')
    else:
        plt.savefig(f'./results/kernel_matrix_visu_r_{args.r}_gamma_{args.gamma}.png')
    wandb.log({'Kernel Matrix': wandb.Image(plt.gcf())})
    plt.close()
    return