#!/usr/bin/env python
"""
patch5.py - Separate MNIST configs by number of classes

Splits MNIST experiments into:
1. Binary classification (2 classes): 0v1, 2v7
2. 4-class classification: 0-3
3. 10-class classification: all digits (separate, slow)

Run from repository root:
    python patch5.py
"""

import json
from pathlib import Path


def create_mnist_configs():
    """Create separate MNIST config files by class count."""

    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Binary classification config
    mnist_binary = {
        "seed": 42,
        "outdir": "outdir",
        "logging": {"level": "info"},
        "dataset": {
            "name": "mnist",
            "root": "data",
            "n_components": 8
        },
        "training": {
            "n_shots": 1000,
            "test_size": 0.5,
            "n_repeats": 10
        },
        "experiments": [
            {
                "classes": [0, 1],
                "max_samples": 80,
                "description": "digits 0 vs 1 (80 samples -> 40 train / 40 test)"
            },
            {
                "classes": [2, 7],
                "max_samples": 80,
                "description": "digits 2 vs 7 (80 samples -> 40 train / 40 test)"
            }
        ]
    }

    # 4-class config
    mnist_4class = {
        "seed": 42,
        "outdir": "outdir",
        "logging": {"level": "info"},
        "dataset": {
            "name": "mnist",
            "root": "data",
            "n_components": 8
        },
        "training": {
            "n_shots": 1000,
            "test_size": 0.5,
            "n_repeats": 10
        },
        "experiments": [
            {
                "classes": [0, 1, 2, 3],
                "max_samples": 160,
                "description": "digits 0-3 (160 samples -> 80 train / 80 test)"
            }
        ]
    }

    # 10-class config (slow, separate)
    mnist_10class = {
        "seed": 42,
        "outdir": "outdir",
        "logging": {"level": "info"},
        "dataset": {
            "name": "mnist",
            "root": "data",
            "n_components": 8
        },
        "training": {
            "n_shots": 1000,
            "test_size": 0.5,
            "n_repeats": 10
        },
        "experiments": [
            {
                "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "max_samples": 400,
                "description": "all digits (400 samples -> 200 train / 200 test)"
            }
        ]
    }

    # Write configs
    (configs_dir / "mnist_binary.json").write_text(json.dumps(mnist_binary, indent=2))
    print(f"  Created: configs/mnist_binary.json")

    (configs_dir / "mnist_4class.json").write_text(json.dumps(mnist_4class, indent=2))
    print(f"  Created: configs/mnist_4class.json")

    (configs_dir / "mnist_10class.json").write_text(json.dumps(mnist_10class, indent=2))
    print(f"  Created: configs/mnist_10class.json")


def create_updated_notebook():
    """Create notebook with separated MNIST sections."""

    NOTEBOOK = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Nearest Centroid Classification on a Trapped Ion Quantum Computer\n",
                    "\n",
                    "**Paper**: Johri et al. (2020) - [arXiv:2012.04145v2](https://arxiv.org/abs/2012.04145)\n",
                    "\n",
                    "Both Cirq and MerLin run as **ideal simulators** (no noise, no error mitigation)."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Setup"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from sklearn import datasets\n",
                    "from sklearn.neighbors import NearestCentroid\n",
                    "from sklearn.decomposition import PCA\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import MinMaxScaler\n",
                    "from sklearn.metrics import accuracy_score\n",
                    "import torchvision\n",
                    "import torchvision.transforms as transforms\n",
                    "from pathlib import Path\n",
                    "\n",
                    "from lib.classifier import QuantumNearestCentroid, MLQuantumNearestCentroid\n",
                    "from lib.synthetic_data import generate_synthetic_data\n",
                    "\n",
                    "np.random.seed(123)\n",
                    "Path('results').mkdir(exist_ok=True)\n",
                    "print(\"Setup complete!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## Helper Functions"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def run_experiment(X, y, classes, n_components=8, n_shots=1000, n_repeats=10,\n",
                    "                   test_size=0.5, max_samples=None, apply_pca=True):\n",
                    "    \"\"\"Run experiment with IDEAL quantum simulators.\"\"\"\n",
                    "    mask = np.isin(y, classes)\n",
                    "    X_filtered, y_filtered = X[mask], y[mask]\n",
                    "    n_available = len(X_filtered)\n",
                    "    n_to_use = min(max_samples, n_available) if max_samples else n_available\n",
                    "    print(f\"  Classes: {classes}, Available: {n_available}, Using: {n_to_use}\")\n",
                    "    \n",
                    "    classical_accs, cirq_accs, merlin_accs = [], [], []\n",
                    "    \n",
                    "    for repeat in range(n_repeats):\n",
                    "        if max_samples and max_samples < n_available:\n",
                    "            X_sub, _, y_sub, _ = train_test_split(X_filtered, y_filtered,\n",
                    "                train_size=max_samples, stratify=y_filtered, random_state=repeat)\n",
                    "        else:\n",
                    "            X_sub, y_sub = X_filtered, y_filtered\n",
                    "        \n",
                    "        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub,\n",
                    "            test_size=test_size, stratify=y_sub, random_state=123+repeat)\n",
                    "        \n",
                    "        if apply_pca and X_train.shape[1] > n_components:\n",
                    "            pca = PCA(n_components=n_components)\n",
                    "            X_train_pca = pca.fit_transform(X_train)\n",
                    "            X_test_pca = pca.transform(X_test)\n",
                    "        else:\n",
                    "            X_train_pca, X_test_pca = X_train, X_test\n",
                    "        \n",
                    "        scaler = MinMaxScaler()\n",
                    "        X_train_scaled = scaler.fit_transform(X_train_pca)\n",
                    "        X_test_scaled = scaler.transform(X_test_pca)\n",
                    "        \n",
                    "        clf = NearestCentroid()\n",
                    "        clf.fit(X_train_pca, y_train)\n",
                    "        classical_accs.append(accuracy_score(y_test, clf.predict(X_test_pca)))\n",
                    "        \n",
                    "        clf_cirq = QuantumNearestCentroid(repetitions=n_shots)\n",
                    "        clf_cirq.fit(X_train_scaled, y_train)\n",
                    "        cirq_accs.append(accuracy_score(y_test, clf_cirq.predict(X_test_scaled)))\n",
                    "        \n",
                    "        clf_ml = MLQuantumNearestCentroid(n=n_components, repetitions=n_shots)\n",
                    "        clf_ml.fit(X_train_scaled, y_train)\n",
                    "        merlin_accs.append(accuracy_score(y_test, clf_ml.predict(X_test_scaled)))\n",
                    "    \n",
                    "    return {'classes': classes, 'n_used': n_to_use, 'n_components': n_components, 'n_shots': n_shots,\n",
                    "            'classical_mean': np.mean(classical_accs), 'classical_std': np.std(classical_accs),\n",
                    "            'cirq_mean': np.mean(cirq_accs), 'cirq_std': np.std(cirq_accs),\n",
                    "            'merlin_mean': np.mean(merlin_accs), 'merlin_std': np.std(merlin_accs)}"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "def plot_error_bars(results_list, title):\n",
                    "    fig, ax = plt.subplots(figsize=(10, 5))\n",
                    "    labels = [f\"Nq={r['n_components']}\\nNc={len(r['classes'])}\\nn={r['n_used']}\" for r in results_list]\n",
                    "    x = np.arange(len(results_list)); width = 0.25\n",
                    "    ax.bar(x-width, [(1-r['classical_mean'])*100 for r in results_list], width,\n",
                    "           yerr=[r['classical_std']*100 for r in results_list], label='Classical', color='#2ecc71', capsize=4)\n",
                    "    ax.bar(x, [(1-r['cirq_mean'])*100 for r in results_list], width,\n",
                    "           yerr=[r['cirq_std']*100 for r in results_list], label='Cirq', color='#3498db', capsize=4)\n",
                    "    ax.bar(x+width, [(1-r['merlin_mean'])*100 for r in results_list], width,\n",
                    "           yerr=[r['merlin_std']*100 for r in results_list], label='MerLin', color='#e74c3c', capsize=4)\n",
                    "    ax.set_ylabel('Error %'); ax.set_title(title); ax.set_xticks(x); ax.set_xticklabels(labels)\n",
                    "    ax.legend(); ax.set_ylim(bottom=0); ax.grid(axis='y', alpha=0.3); plt.tight_layout()\n",
                    "    return fig"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["---\n", "## Synthetic Data (Figure 8)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "synthetic_configs = [\n",
                    "    {'n_clusters': 2, 'n_dimensions': 4, 'n_shots': 100},\n",
                    "    {'n_clusters': 4, 'n_dimensions': 4, 'n_shots': 500},\n",
                    "    {'n_clusters': 2, 'n_dimensions': 8, 'n_shots': 1000},\n",
                    "    {'n_clusters': 4, 'n_dimensions': 8, 'n_shots': 1000},\n",
                    "]\n",
                    "synthetic_results = []\n",
                    "\n",
                    "for cfg in synthetic_configs:\n",
                    "    print(f\"Nq={cfg['n_dimensions']}, Nc={cfg['n_clusters']}\")\n",
                    "    X, y, _ = generate_synthetic_data(n_clusters=cfg['n_clusters'], n_dimensions=cfg['n_dimensions'],\n",
                    "        n_points_per_cluster=10, min_centroid_distance=0.3, gaussian_variance=0.05, seed=123)\n",
                    "    result = run_experiment(X, y, list(range(cfg['n_clusters'])), n_components=cfg['n_dimensions'],\n",
                    "        n_shots=cfg['n_shots'], apply_pca=False)\n",
                    "    synthetic_results.append(result)\n",
                    "    print(f\"  Classical={result['classical_mean']*100:.1f}%, Cirq={result['cirq_mean']*100:.1f}%, MerLin={result['merlin_mean']*100:.1f}%\\n\")\n",
                    "\n",
                    "fig = plot_error_bars(synthetic_results, \"Figure 8: Synthetic Data\")\n",
                    "plt.savefig('results/fig8_synthetic.png', dpi=150); plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["---\n", "## IRIS (Figure 9)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "iris = datasets.load_iris()\n",
                    "iris_results = []\n",
                    "for n_shots in [100, 500, 1000]:\n",
                    "    print(f\"IRIS Ns={n_shots}\")\n",
                    "    result = run_experiment(iris.data, iris.target, [0,1,2], n_components=4,\n",
                    "        n_shots=n_shots, max_samples=150, apply_pca=False)\n",
                    "    iris_results.append(result)\n",
                    "    print(f\"  Cirq={result['cirq_mean']*100:.1f}%, MerLin={result['merlin_mean']*100:.1f}%\\n\")\n",
                    "\n",
                    "fig = plot_error_bars(iris_results, \"Figure 9: IRIS\")\n",
                    "plt.savefig('results/fig9_iris.png', dpi=150); plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["---\n", "## MNIST - Load Data"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"Loading MNIST...\")\n",
                    "transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])\n",
                    "mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)\n",
                    "X_mnist, y_mnist = mnist.data.numpy().reshape(len(mnist), -1), mnist.targets.numpy()\n",
                    "print(f\"MNIST: {X_mnist.shape[0]} samples, {X_mnist.shape[1]} features\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### MNIST Binary Classification (2 classes)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "mnist_binary_results = []\n",
                    "\n",
                    "for classes, desc in [([0, 1], '0 vs 1'), ([2, 7], '2 vs 7')]:\n",
                    "    print(f\"MNIST Binary: {desc}\")\n",
                    "    result = run_experiment(X_mnist, y_mnist, classes=classes, n_components=8,\n",
                    "        n_shots=1000, max_samples=80, apply_pca=True)\n",
                    "    result['description'] = desc\n",
                    "    mnist_binary_results.append(result)\n",
                    "    print(f\"  Classical={result['classical_mean']*100:.1f}%, Cirq={result['cirq_mean']*100:.1f}%, MerLin={result['merlin_mean']*100:.1f}%\\n\")\n",
                    "\n",
                    "fig = plot_error_bars(mnist_binary_results, \"Figure 11a: MNIST Binary Classification\")\n",
                    "plt.savefig('results/fig11a_mnist_binary.png', dpi=150); plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### MNIST 4-Class Classification"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"MNIST 4-class: digits 0-3\")\n",
                    "mnist_4class_result = run_experiment(X_mnist, y_mnist, classes=[0,1,2,3], n_components=8,\n",
                    "    n_shots=1000, max_samples=160, apply_pca=True)\n",
                    "print(f\"  Classical={mnist_4class_result['classical_mean']*100:.1f}%, Cirq={mnist_4class_result['cirq_mean']*100:.1f}%, MerLin={mnist_4class_result['merlin_mean']*100:.1f}%\")\n",
                    "\n",
                    "fig = plot_error_bars([mnist_4class_result], \"Figure 11b: MNIST 4-Class\")\n",
                    "plt.savefig('results/fig11b_mnist_4class.png', dpi=150); plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["### MNIST 10-Class Classification (SLOW - skip if needed)"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# WARNING: This cell is slow! Skip if needed.\n",
                    "RUN_10CLASS = False  # Set to True to run\n",
                    "\n",
                    "if RUN_10CLASS:\n",
                    "    print(\"MNIST 10-class: all digits (this is slow!)\")\n",
                    "    mnist_10class_result = run_experiment(X_mnist, y_mnist, classes=list(range(10)), n_components=8,\n",
                    "        n_shots=1000, max_samples=400, apply_pca=True)\n",
                    "    print(f\"  Classical={mnist_10class_result['classical_mean']*100:.1f}%, Cirq={mnist_10class_result['cirq_mean']*100:.1f}%, MerLin={mnist_10class_result['merlin_mean']*100:.1f}%\")\n",
                    "    \n",
                    "    fig = plot_error_bars([mnist_10class_result], \"Figure 11c: MNIST 10-Class\")\n",
                    "    plt.savefig('results/fig11c_mnist_10class.png', dpi=150); plt.show()\n",
                    "else:\n",
                    "    print(\"Skipping 10-class MNIST. Set RUN_10CLASS = True to run.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["---\n", "## Summary\n", "\n", "Both Cirq and MerLin run as ideal simulators. Results should match within shot noise."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(\"=\" * 50)\n",
                    "print(\"RESULTS SUMMARY\")\n",
                    "print(\"=\" * 50)\n",
                    "\n",
                    "print(\"\\n--- Synthetic ---\")\n",
                    "for r in synthetic_results:\n",
                    "    print(f\"Nq={r['n_components']}, Nc={len(r['classes'])}: C={r['classical_mean']*100:.1f}%, Cirq={r['cirq_mean']*100:.1f}%, ML={r['merlin_mean']*100:.1f}%\")\n",
                    "\n",
                    "print(\"\\n--- IRIS ---\")\n",
                    "for r in iris_results:\n",
                    "    print(f\"Ns={r['n_shots']}: C={r['classical_mean']*100:.1f}%, Cirq={r['cirq_mean']*100:.1f}%, ML={r['merlin_mean']*100:.1f}%\")\n",
                    "\n",
                    "print(\"\\n--- MNIST Binary ---\")\n",
                    "for r in mnist_binary_results:\n",
                    "    print(f\"{r['description']}: C={r['classical_mean']*100:.1f}%, Cirq={r['cirq_mean']*100:.1f}%, ML={r['merlin_mean']*100:.1f}%\")\n",
                    "\n",
                    "print(\"\\n--- MNIST 4-class ---\")\n",
                    "print(f\"0-3: C={mnist_4class_result['classical_mean']*100:.1f}%, Cirq={mnist_4class_result['cirq_mean']*100:.1f}%, ML={mnist_4class_result['merlin_mean']*100:.1f}%\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("notebook.ipynb", "w") as f:
        json.dump(NOTEBOOK, f, indent=1)
    print(f"  Created: notebook.ipynb")


def main():
    print("=" * 60)
    print("PATCH 5: Separate MNIST configs by class count")
    print("=" * 60)

    print("\nCreating separate MNIST config files...")
    create_mnist_configs()

    print("\nCreating updated notebook...")
    create_updated_notebook()

    print("\n" + "=" * 60)
    print("PATCH 5 COMPLETE")
    print("=" * 60)
    print("""
Config files created:
  configs/mnist_binary.json  - 2-class: [0,1] and [2,7], 80 samples each
  configs/mnist_4class.json  - 4-class: [0,1,2,3], 160 samples
  configs/mnist_10class.json - 10-class: all digits, 400 samples (slow)

Notebook changes:
  - MNIST split into separate sections:
    1. Binary classification (fast)
    2. 4-class classification (fast)
    3. 10-class classification (slow, disabled by default)

  - 10-class has RUN_10CLASS = False flag to skip

CLI usage:
  python implementation.py --config configs/mnist_binary.json   # Fast
  python implementation.py --config configs/mnist_4class.json   # Fast
  python implementation.py --config configs/mnist_10class.json  # Slow
""")


if __name__ == "__main__":
    main()