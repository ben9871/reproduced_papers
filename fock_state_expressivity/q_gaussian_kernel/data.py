import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.model_selection import train_test_split

# For training quantum Gaussian kernel sampler

def get_data_x():
    """
    Get x values of data (1-D features), x / pi for visualization and delta for input to quantum circuit.
    """
    # We want x form -pi to pi
    x = np.linspace(-np.pi, np.pi, num=int(2 * np.pi / 0.05) + 1)
    # We will use x_on_pi for visualization
    x_on_pi = x / np.pi
    # delta will be the input to our quantum model
    delta = (x - 0) ** 2
    return x, x_on_pi, delta

def target_function(delta, sigma=1.0):
    """
    Get target value for delta input (regression)
    :param delta: squared distance between two points
    :param sigma: standard deviation used
    :return: target value for regression
    """
    return np.exp(-delta/(2*sigma*sigma))

def visu_target_functions(x_on_pi, ys):
    # Plot using matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    i = 0
    j = 0
    sigmas = [1.00, 0.50, 0.33, 0.25]

    for y, sigma in zip(ys, sigmas):
        axis = axs[i // 2][j % 2]
        axis.scatter(x_on_pi, y, label=f"$\sigma$={sigma}", s=10, color="k")
        axis.set_xlabel('x / pi')
        axis.set_ylabel('y')
        axis.grid(True)
        axis.legend(loc="upper right")
        i += 1
        j += 1

    fig.suptitle("Gaussian function for different standard deviations", fontsize=14)
    plt.savefig("./target_gaussian_functions.png")
    plt.show()
    plt.clf()
    return

# For classification

def get_circle(num_samples, noise=0.1):
    X, y = make_circles(
        n_samples=num_samples,  # number of data points
        noise=noise,
        random_state=42
    )
    return X, y


def get_moon(num_samples, noise=0.2):
    X, y = make_moons(
        n_samples=num_samples,  # number of data points
        noise=noise,
        random_state=42
    )
    return X, y


def get_blobs(num_samples, centers=2):
    X, y = make_blobs(
        n_samples=num_samples,
        centers=centers,
        cluster_std=4.0,
        random_state=42
    )
    return X, y


def get_visual_sample(x, y, x_test, y_test, title):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].scatter(x[0][:, 0], x[0][:, 1], c=y[0], cmap='bwr', marker='o', label='Train set')
    axs[0].scatter(x_test[0][:, 0], x_test[0][:, 1], c=y_test[0], cmap='bwr', marker='x', label='Test set')
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")
    # Create dummy legend handles with gray color
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', label='Train set'),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', label='Test set')
    ]
    axs[0].legend(handles=legend_elements)
    #axs[0].legend()
    axs[1].scatter(x[1][:, 0], x[1][:, 1], c=y[1], cmap='bwr', marker='o', label='Train set')
    axs[1].scatter(x_test[1][:, 0], x_test[1][:, 1], c=y_test[1], cmap='bwr', marker='x', label='Test set')
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")
    # Create dummy legend handles with gray color
    axs[1].legend(handles=legend_elements)
    #axs[1].legend()
    axs[2].scatter(x[2][:, 0], x[2][:, 1], c=y[2], cmap='bwr', marker='o', label='Train set')
    axs[2].scatter(x_test[2][:, 0], x_test[2][:, 1], c=y_test[2], cmap='bwr', marker='x', label='Test set')
    axs[2].set_xlabel("x1")
    axs[2].set_ylabel("x2")
    # Create dummy legend handles with gray color
    axs[2].legend(handles=legend_elements)
    #axs[2].legend()
    axs[0].title.set_text(title[0])
    axs[1].title.set_text(title[1])
    axs[2].title.set_text(title[2])



    plt.savefig("./results/data_visualization.png")  # To save figure locally
    '''plt.show()
    plt.clf()'''
    return

def prepare_data(scaling_factor=0.65):
    """Standardization, type changing and splitting of the data for preparation"""
    x_circ, y_circ = get_circle(200)
    x_moon, y_moon = get_moon(200)
    x_blob, y_blob = get_blobs(200)

    x_circ_train, x_circ_test, y_circ_train, y_circ_test = train_test_split(x_circ, y_circ, test_size=0.4, random_state=42)

    # Convert data to PyTorch tensors
    x_circ_train = torch.FloatTensor(x_circ_train)
    y_circ_train = torch.FloatTensor(y_circ_train)
    x_circ_test = torch.FloatTensor(x_circ_test)
    y_circ_test = torch.FloatTensor(y_circ_test)

    scaler = StandardScaler()
    x_circ_train = torch.FloatTensor(scaler.fit_transform(x_circ_train)) * scaling_factor
    x_circ_test = torch.FloatTensor(scaler.transform(x_circ_test)) * scaling_factor

    print(f"Circular training set: {x_circ_train.shape[0]} samples, {x_circ_train.shape[1]} features")
    print(f"Circular test set: {x_circ_test.shape[0]} samples, {x_circ_test.shape[1]} features")

    x_moon_train, x_moon_test, y_moon_train, y_moon_test = train_test_split(x_moon, y_moon, test_size=0.4, random_state=42)

    # Convert data to PyTorch tensors
    x_moon_train = torch.FloatTensor(x_moon_train)
    y_moon_train = torch.FloatTensor(y_moon_train)
    x_moon_test = torch.FloatTensor(x_moon_test)
    y_moon_test = torch.FloatTensor(y_moon_test)

    scaler = StandardScaler()
    x_moon_train = torch.FloatTensor(scaler.fit_transform(x_moon_train)) * scaling_factor
    x_moon_test = torch.FloatTensor(scaler.transform(x_moon_test)) * scaling_factor

    print(f"Moon training set: {x_moon_train.shape[0]} samples, {x_moon_train.shape[1]} features")
    print(f"Moon test set: {x_moon_test.shape[0]} samples, {x_moon_test.shape[1]} features")

    x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob, test_size=0.4, random_state=42)

    # Convert data to PyTorch tensors
    x_blob_train = torch.FloatTensor(x_blob_train)
    y_blob_train = torch.FloatTensor(y_blob_train)
    x_blob_test = torch.FloatTensor(x_blob_test)
    y_blob_test = torch.FloatTensor(y_blob_test)

    scaler = StandardScaler()
    x_blob_train = torch.FloatTensor(scaler.fit_transform(x_blob_train)) * scaling_factor
    x_blob_test = torch.FloatTensor(scaler.transform(x_blob_test)) * scaling_factor

    print(f"Blob training set: {x_blob_train.shape[0]} samples, {x_blob_train.shape[1]} features")
    print(f"Blob test set: {x_blob_test.shape[0]} samples, {x_blob_test.shape[1]} features")

    x_train = [x_circ_train, x_moon_train, x_blob_train]
    x_test = [x_circ_test, x_moon_test, x_blob_test]
    y_train = [y_circ_train, y_moon_train, y_blob_train]
    y_test = [y_circ_test, y_moon_test, y_blob_test]

    return x_train, x_test, y_train, y_test