#!/usr/bin/env python3
"""Plot per-epoch comparisons laid out as a grid (runs × epochs).

Usage:
    # compare 3 runs at epochs 1, 15, 30, 100 in a 3×4 grid
    python -m utils.aggregate_plots --runs RUN1:Label1 RUN2:Label2 RUN3:Label3 \
        --epochs 1,15,30,100 --out outdir/aggregate.png --width 4 --height 3

Each RUNx should be a run directory containing ``simulation_data_e{epoch}.npz``.
We create one subplot per (run, epoch) pair, aligned so columns share the same
epoch and rows share the same run. Every panel shows the ground truth together
with the selected run's prediction at that epoch.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_npz(run_dir: Path, epoch: int):
    p = run_dir / f"simulation_data_e{epoch}.npz"
    d = np.load(p)
    return d["y"], d["y_pred"], int(d["n_train"]) if "n_train" in d else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate QLSTM/LSTM snapshots into one plot")
    ap.add_argument("--runs", nargs="+", help="List of RUN_DIR:LABEL (LABEL recommended for a readable legend)")
    ap.add_argument(
        "--epochs", type=str, required=True,
        help="Comma-separated list of epochs (e.g., 30,60,100). Single epoch supported (e.g., 100)"
    )
    ap.add_argument("--out", type=str, required=True, help="Output image path (suffix defines format)")
    ap.add_argument("--width", type=float, default=6.0, help="Width of each subplot in inches")
    ap.add_argument("--height", type=float, default=None, help="Height of each subplot in inches (default: width * 0.6)")
    ap.add_argument("--fmt", type=str, default=None, help="Force format (png|pdf). Default from suffix")
    args = ap.parse_args(argv)

    if not args.runs:
        raise ValueError("--runs must be provided with at least one entry")

    # Prepare the list of epochs
    try:
        epochs = [int(x) for x in args.epochs.split(",") if str(x).strip()]
    except Exception as e:
        raise ValueError(f"Format invalide pour --epochs: {args.epochs!r}") from e

    parsed_runs: list[tuple[Path, str]] = []
    for spec in args.runs:
        path_label = spec.split(":", 1)
        run_path = Path(path_label[0])
        label = path_label[1] if len(path_label) > 1 else run_path.name
        parsed_runs.append((run_path, label))

    if not parsed_runs:
        raise ValueError("No runs provided")

    n_epochs = len(epochs)
    n_runs = len(parsed_runs)
    subplot_height = args.height if args.height is not None else args.width * 0.6
    fig_width = args.width * n_epochs
    fig_height = subplot_height * n_runs
    fig, axes = plt.subplots(n_runs, n_epochs, figsize=(fig_width, fig_height), sharex=False, sharey=True, squeeze=False)

    vline_x = None
    gt = None

    for row_idx, (run_path, label) in enumerate(parsed_runs):
        color = "C0"
        for col_idx, ep in enumerate(epochs):
            ax = axes[row_idx, col_idx]
            y, y_pred, n_train = load_npz(run_path, ep)
            if gt is None:
                gt = y
                vline_x = n_train
            elif not np.allclose(gt, y):
                raise ValueError(f"Ground truth mismatch between runs for epoch {ep}: {run_path}")

            ax.plot(gt, "--", color="orange", linewidth=2, label="Ground Truth" if row_idx == 0 and col_idx == 0 else "")
            ax.plot(y_pred, color=color, linewidth=2, label=label if col_idx == 0 else "")

            if vline_x is not None:
                ax.axvline(x=vline_x, c="r", linestyle="--")

            if row_idx == 0:
                ax.set_title(f"Epoch {ep}")
            ax.set_xlabel("Timestep")
            if col_idx == 0:
                ax.set_ylabel(label)
            else:
                ax.set_ylabel("")

            if col_idx == 0:
                handles, labels = ax.get_legend_handles_labels()
                filtered = []
                seen: set[str] = set()
                for h, l in zip(handles, labels):
                    if l and l not in seen:
                        filtered.append((h, l))
                        seen.add(l)
                if filtered:
                    ax.legend(*zip(*filtered), loc="lower left")

    fig.tight_layout()
    out_path = Path(args.out)
    fmt = args.fmt or out_path.suffix.lstrip(".")
    fig.savefig(out_path, format=fmt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
