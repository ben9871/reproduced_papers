#!/usr/bin/env python3
"""Compare loss curves across multiple QLSTM runs.

Example:
    python -m utils.compare_losses \
        --runs outdir/run_20251116-195040:Photonic \
               outdir/run_20251116-195059:Gate \
        --metric test \
        --out outdir/loss_comparison_20251116.png

Each run directory must contain a ``losses.csv`` file as written by the
training script. The ``--metric`` flag controls whether we plot the train or
test loss (default: test).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib

# Use a non-interactive backend so the script can run headless.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def read_losses(csv_path: Path) -> tuple[list[int], list[float], list[float]]:
    epochs: list[int] = []
    train: list[float] = []
    test: list[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_mse"]))
            test.append(float(row["test_mse"]))
    return epochs, train, test


def parse_run_specs(specs: Sequence[str]) -> list[tuple[Path, str]]:
    parsed: list[tuple[Path, str]] = []
    for spec in specs:
        path_label = spec.split(":", 1)
        run_path = Path(path_label[0])
        label = path_label[1] if len(path_label) > 1 else run_path.name
        parsed.append((run_path, label))
    return parsed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare train/test loss curves across runs")
    ap.add_argument("--runs", nargs="+", required=True, help="List of RUN_DIR[:LABEL]")
    ap.add_argument("--metric", choices=["train", "test"], default="test", help="Loss curve to plot")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--width", type=float, default=6.0, help="Figure width in inches")
    ap.add_argument("--height", type=float, default=3.5, help="Figure height in inches")
    args = ap.parse_args(argv)

    runs = parse_run_specs(args.runs)
    if not runs:
        raise ValueError("No runs provided")

    plt.figure(figsize=(args.width, args.height))
    metric_key = "test" if args.metric == "test" else "train"

    for run_path, label in runs:
        csv_path = run_path / "losses.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing losses.csv in run directory: {run_path}")
        epochs, train_losses, test_losses = read_losses(csv_path)
        series = test_losses if metric_key == "test" else train_losses
        plt.plot(epochs, series, label=label, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Test MSE" if metric_key == "test" else "Train MSE")
    plt.title(f"{args.metric.capitalize()} loss comparison")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out_path = Path(args.out)
    fmt = out_path.suffix.lstrip(".") or "png"
    plt.savefig(out_path, format=fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
