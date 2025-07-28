"""
Dataset generation and visualization utilities for VQC classification experiments.

This module provides functions to generate synthetic datasets (linear, circular, moon-shaped)
and visualize them for binary classification tasks.
"""

from sklearn.datasets import make_classification, make_circles, make_moons
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import os

def get_linear(num_samples, num_features, class_sep=1.0):
    """
    Generate linearly separable binary classification dataset.
    
    Args:
        num_samples (int): Number of data points to generate
        num_features (int): Number of input features (typically 2 for visualization)
        class_sep (float): Class separation factor (higher = more separable)
        
    Returns:
        tuple: (X, y) where X is features array and y is labels array
    """
    X, y = make_classification(
        n_samples=num_samples,
        n_features=num_features,
        n_informative=2,  # Both features carry class information
        n_redundant=0,
        n_clusters_per_class=1,
        n_classes=2,
        random_state=42,
        class_sep=class_sep
    )
    return X, y


def get_circle(num_samples, noise=0.1):
    """
    Generate circular binary classification dataset.
    
    Args:
        num_samples (int): Number of data points to generate
        noise (float): Standard deviation of Gaussian noise added to data
        
    Returns:
        tuple: (X, y) where X is features array and y is labels array
    """
    X, y = make_circles(
        n_samples=num_samples,
        noise=noise,
        random_state=42
    )
    return X, y


def get_moon(num_samples, noise=0.2):
    """
    Generate moon-shaped binary classification dataset.
    
    Args:
        num_samples (int): Number of data points to generate
        noise (float): Standard deviation of Gaussian noise added to data
        
    Returns:
        tuple: (X, y) where X is features array and y is labels array
    """
    X, y = make_moons(
        n_samples=num_samples,
        noise=noise,
        random_state=42
    )
    return X, y


def get_visual_sample(x, y, title="Data visualization"):
    """
    Create and save a scatter plot visualization of the dataset.
    
    Args:
        x (numpy.ndarray): Feature data with shape (n_samples, 2)
        y (numpy.ndarray): Binary labels (0 or 1)
        title (str): Title for the plot and filename
    """
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', edgecolor='k')
    plt.xlabel("x1")
    plt.ylabel("x2")
    new_title = "of\n".join(title.split("of"))
    plt.title(new_title, size=16)
    legend_elements = [
        Patch(facecolor='blue', edgecolor='k', label='label = 0'),
        Patch(facecolor='red', edgecolor='k', label='label = 1')
    ]
    plt.legend(handles=legend_elements)
    plt.savefig("./results/" + title + '.png')
    plt.clf()


def save_dataset_locally(path, X_train, X_test, y_train, y_test):
    """
    Save train/test dataset splits to local file.
    
    Args:
        path (str): File path to save the dataset
        X_train (torch.Tensor): Training features
        X_test (torch.Tensor): Test features
        y_train (torch.Tensor): Training labels
        y_test (torch.Tensor): Test labels
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }, path)