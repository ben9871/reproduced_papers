from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data.datasets import DATASET_ORDER, prepare_datasets
from lib.training import ExperimentArgs, summarize_results, train_model_multiple_runs
from utils.plotting import (
    plot_dataset_samples,
    plot_decision_boundary,
    plot_training_metrics,
)

MODEL_TYPE_PRESETS: dict[str, tuple[str, list[int]]] = {
    "vqc_100": ("vqc", [1, 0, 0]),
    "vqc_111": ("vqc", [1, 1, 1]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Photonic VQC classification reproduction."
    )
    parser.add_argument(
        "--config", default="configs/defaults.json", help="Path to JSON config."
    )
    parser.add_argument("--seed", type=int, help="Override RNG seed.")
    parser.add_argument("--outdir", help="Override output base directory.")
    parser.add_argument(
        "--model-type",
        choices=[
            "vqc",
            "vqc_100",
            "vqc_111",
            "mlp_wide",
            "mlp_deep",
            "svm_lin",
            "svm_rbf",
        ],
        help="Experiment model type.",
    )
    parser.add_argument("--device", default=None, help="Torch device string.")
    parser.add_argument(
        "--visualize-data", action="store_true", help="Force dataset scatter plots."
    )
    parser.add_argument(
        "--skip-boundaries", action="store_true", help="Skip decision boundary plots."
    )
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        help="Enable Weights & Biases logging regardless of config.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_run_directory(base: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _normalize_model_type(label: str) -> tuple[str, list[int] | None]:
    base, preset = MODEL_TYPE_PRESETS.get(label, (label, None))
    return base, preset.copy() if preset is not None else None


def build_args(config: dict[str, Any], cli: argparse.Namespace) -> ExperimentArgs:
    model_cfg = config["model"]
    training_cfg = config["training"]
    logging_cfg = config.get("logging", {})
    betas = training_cfg.get("betas", [0.9, 0.999])

    args = ExperimentArgs(
        m=model_cfg.get("num_modes", 3),
        input_size=model_cfg.get("input_size", 2),
        initial_state=model_cfg.get("initial_state", [1, 1, 1]),
        activation=model_cfg.get("activation", "none"),
        no_bunching=model_cfg.get("no_bunching", False),
        num_runs=training_cfg.get("num_runs", 5),
        n_epochs=training_cfg.get("epochs", 150),
        batch_size=training_cfg.get("batch_size", 30),
        lr=training_cfg.get("learning_rate", 0.02),
        alpha=training_cfg.get("alpha", 0.0),
        betas=(betas[0], betas[1]),
        circuit=model_cfg.get("circuit", "bs_mesh"),
        scale_type=model_cfg.get("scale_type", "learned"),
        regu_on=model_cfg.get("regularization_target"),
        log_wandb=cli.log_wandb or logging_cfg.get("log_wandb", False),
        wandb_project=logging_cfg.get("wandb_project", "vqc_reproduction"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        device=cli.device or config.get("device", "cpu"),
    )
    requested_model = cli.model_type or config["experiment"].get("model_type", "vqc")
    base_model, preset_state = _normalize_model_type(requested_model)
    if preset_state is not None:
        args.initial_state = preset_state
    args.requested_model_type = requested_model
    args.set_model_type(base_model)
    return args


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    config.setdefault("experiment", {})

    if cli.seed is not None:
        config["seed"] = cli.seed
    config.setdefault("seed", 1337)
    set_seed(int(config["seed"]))

    outdir = Path(cli.outdir or config.get("outdir", "results"))
    run_dir = prepare_run_directory(outdir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config.get("data", {})
    datasets = prepare_datasets(data_cfg)

    if data_cfg.get("visualize", False) or cli.visualize_data:
        dataset_fig_dir = figures_dir / "datasets"
        for name in DATASET_ORDER:
            if name not in datasets:
                continue
            payload = datasets[name]
            plot_dataset_samples(
                torch.cat((payload["x_train"], payload["x_test"])),
                torch.cat((payload["y_train"], payload["y_test"])),
                f"{name.title()} dataset",
                dataset_fig_dir / f"{name}_scatter.png",
            )

    args = build_args(config, cli)

    circuit_dir = (figures_dir / "circuits").as_posix()
    results, best_models = train_model_multiple_runs(
        args.model_type,
        args,
        datasets,
        circuit_dir=circuit_dir,
    )

    summary = summarize_results(results, args)
    (run_dir / "summary.txt").write_text(summary)

    metrics = {
        dataset: {
            "final_test_accs": [float(run["final_test_acc"]) for run in data["runs"]],
            "avg_final_test_acc": float(data["avg_final_test_acc"]),
        }
        for dataset, data in results.items()
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    outputs_cfg = config.get("outputs", {})
    if outputs_cfg.get("save_training_curves", True):
        plot_training_metrics(
            results, figures_dir / "training_metrics.png", dataset_order=DATASET_ORDER
        )

    if outputs_cfg.get("save_decision_boundaries", True) and not cli.skip_boundaries:
        boundary_dir = figures_dir / "decision_boundaries"
        for entry in best_models:
            plot_decision_boundary(
                entry,
                boundary_dir / f"{entry['dataset']}_{args.requested_model_type}.png",
            )

    print(f"Artifacts saved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
