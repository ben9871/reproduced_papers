#!/usr/bin/env python
"""
run_full_paper_reproduction.py

Executes ALL experiments from the paper:
1. Synthetic data (Fig 8, Tables I & II)
2. IRIS dataset (Fig 9, Fig 10)
3. MNIST dataset (Fig 11, Fig 12)
4. Noise model validation (Fig 13)

Run from repository root:
    python run_full_paper_reproduction.py
"""

import json
import sys
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent))

from lib.runner import main, set_seed
from lib.synthetic_data import generate_synthetic_data, generate_paper_datasets
from lib.visualization import (
    plot_combined_figure,
    plot_confusion_matrices_comparison,
    create_label_comparison_table,
    plot_iris_pca_scatter,
    plot_iris_comparison,
    plot_c_exp_vs_c_sim,
    plot_synthetic_results,
)


def run_synthetic_experiments():
    """Run all synthetic data experiments (Fig 8, Tables I, II)."""
    print("\n" + "="*60)
    print("SYNTHETIC DATA EXPERIMENTS")
    print("="*60)

    configs = [
        "configs/synthetic_4q_2c.json",
        "configs/synthetic_4q_4c.json",
        "configs/synthetic_8q_2c.json",
        "configs/synthetic_8q_4c.json",
    ]

    for config_path in configs:
        if Path(config_path).exists():
            print(f"\nRunning: {config_path}")
            cfg = json.loads(Path(config_path).read_text())
            set_seed(cfg.get("seed", 123))
            run_dir = main(cfg)
            print(f"Results saved to: {run_dir}")
        else:
            print(f"Config not found: {config_path}")


def run_iris_experiments():
    """Run all IRIS experiments (Fig 9, Fig 10)."""
    print("\n" + "="*60)
    print("IRIS DATASET EXPERIMENTS")
    print("="*60)

    config_path = "configs/iris_3class.json"
    if Path(config_path).exists():
        cfg = json.loads(Path(config_path).read_text())
        set_seed(cfg.get("seed", 123))
        run_dir = main(cfg)
        print(f"Results saved to: {run_dir}")
    else:
        # Fallback to default iris config
        config_path = "configs/iris_config.json"
        if Path(config_path).exists():
            cfg = json.loads(Path(config_path).read_text())
            set_seed(cfg.get("seed", 123))
            run_dir = main(cfg)
            print(f"Results saved to: {run_dir}")


def run_mnist_experiments():
    """Run all MNIST experiments (Fig 11, Fig 12)."""
    print("\n" + "="*60)
    print("MNIST DATASET EXPERIMENTS")
    print("="*60)

    config_path = "configs/mnist_config.json"
    if Path(config_path).exists():
        cfg = json.loads(Path(config_path).read_text())
        set_seed(cfg.get("seed", 123))
        run_dir = main(cfg)
        print(f"Results saved to: {run_dir}")


def main_reproduction():
    """Run complete paper reproduction."""
    print("="*60)
    print("FULL PAPER REPRODUCTION")
    print("Johri et al. (2020) - Nearest Centroid Classification")
    print("on a Trapped Ion Quantum Computer")
    print("="*60)

    run_synthetic_experiments()
    run_iris_experiments()
    run_mnist_experiments()

    print("\n" + "="*60)
    print("REPRODUCTION COMPLETE")
    print("="*60)
    print("\nGenerated figures should match:")
    print("  - Fig 8: Synthetic data results")
    print("  - Fig 9: IRIS data results")
    print("  - Fig 10: IRIS 2D PCA visualization")
    print("  - Fig 11: MNIST data results")
    print("  - Fig 12: MNIST confusion matrices")
    print("  - Fig 13: c_exp vs c_sim (noise model validation)")
    print("  - Tables I, II: Label comparison tables")


if __name__ == "__main__":
    main_reproduction()
