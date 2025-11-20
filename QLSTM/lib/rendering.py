"""Rendering utilities for QLSTM reproduction.

Adapted from Quantum_Long_Short_Term_Memory (MIT License) with simplifications.
"""
from __future__ import annotations

import os
import pickle
import time
from collections.abc import Sequence

import matplotlib

# Force a non-interactive backend so training runs do not spawn GUI windows.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def timestamp():
    return time.strftime('%Y%m%d-%H%M%S')

def ensure_dir(path:str):
    os.makedirs(path, exist_ok=True)

def save_losses_plot(losses: Sequence[float], out_dir: str, prefix: str = 'loss'):
    ensure_dir(out_dir)
    plt.figure(figsize=(5,3))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.tight_layout()
    # Deterministic name within run dir (directory already has timestamp)
    fname = os.path.join(out_dir, f'{prefix}_plot.png')
    plt.savefig(fname)
    plt.close()
    return fname

def save_simulation_plot(
    y_true,
    y_pred,
    out_dir: str,
    prefix: str = 'simulation',
    vline_x: int | None = None,
    *,
    width: float | None = None,
    title: str | None = None,
):
    ensure_dir(out_dir)
    w = float(width) if width else 6.0
    plt.figure(figsize=(w, 3))
    # Style per paper: Ground Truth dashed orange, Prediction solid blue
    plt.plot(y_true, label='Ground Truth', color='C1', linestyle='--', linewidth=1.5)
    plt.plot(y_pred, label='Prediction', color='C0', linestyle='-', linewidth=1.5)
    if vline_x is not None:
        plt.axvline(vline_x, color='red', linestyle=':', linewidth=1.2)
        plt.text(vline_x, plt.ylim()[1], ' train end', color='red', fontsize=8, va='top', ha='left')
    if title:
        plt.title(title, fontsize=10)
    plt.legend(loc='lower left')
    plt.tight_layout()
    # Deterministic name; callers provide epoch in prefix when needed, e.g. 'simulation_e30'
    fname = os.path.join(out_dir, f'{prefix}.png')
    plt.savefig(fname)
    plt.close()
    return fname

def save_pickle(obj, out_dir: str, name: str):
    ensure_dir(out_dir)
    # Stable filename; timestamp is carried by the run directory already
    path = os.path.join(out_dir, f'{name}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return path
