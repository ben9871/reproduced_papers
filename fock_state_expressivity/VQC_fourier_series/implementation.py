from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data.fourier_series import generate_dataset
from lib.training import summarize_results, train_models_multiple_runs
from lib.vqc import VQCFactory
from utils.plotting import plot_learned_functions, plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQC Fourier-series expressivity experiment."
    )
    parser.add_argument(
        "--config", default="configs/defaults.json", help="Path to JSON config file."
    )
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--outdir", help="Override output directory.")
    parser.add_argument(
        "--device", default="cpu", help="Device string for torch (default: cpu)."
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_run_directory(base: str | Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.seed is not None:
        config["seed"] = args.seed
    if args.outdir:
        config["outdir"] = args.outdir
    config.setdefault("outdir", "results")
    config.setdefault("seed", 1337)
    config.setdefault("device", args.device or "cpu")

    set_seed(int(config["seed"]))
    device = torch.device(config.get("device", "cpu"))

    dataset = generate_dataset(config["data"])
    x_tensor = dataset["x"]
    y_tensor = dataset["y"]

    model_factory = VQCFactory(config["model"])
    training_cfg = config["training"]
    initial_states = training_cfg["initial_states"]
    plotting_cfg = config.get("plotting", {})
    color_palette = plotting_cfg.get("colors") or None

    results, best_models = train_models_multiple_runs(
        model_factory,
        initial_states,
        x_tensor,
        y_tensor,
        training_cfg,
        device=device,
        colors=color_palette,
    )

    outdir = prepare_run_directory(config["outdir"])
    (outdir / "figures").mkdir(exist_ok=True)

    summary_text = summarize_results(results)
    (outdir / "summary.txt").write_text(summary_text)

    metrics = {
        label: {
            "final_mses": [run["train_mses"][-1] for run in data["runs"]],
            "loss_curves": data["runs"],
            "initial_state": data["initial_state"],
        }
        for label, data in results.items()
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    training_plot_path = outdir / "figures" / "training_curves.png"
    learned_plot_path = outdir / "figures" / "learned_vs_target.png"

    plot_training_curves(results, save_path=training_plot_path, show=args.show_plots)
    plot_learned_functions(
        best_models,
        x_tensor,
        y_tensor,
        save_path=learned_plot_path,
        show=args.show_plots,
    )

    print(summary_text)
    print(f"Artifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
