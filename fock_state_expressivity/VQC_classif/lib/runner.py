"""Runtime entrypoints for the VQC classification reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from data.datasets import DATASET_ORDER, prepare_datasets
from utils.plotting import (
    plot_dataset_samples,
    plot_decision_boundary,
    plot_training_metrics,
)

from lib.training import ExperimentArgs, summarize_results, train_model_multiple_runs

LOGGER = logging.getLogger(__name__)

MODEL_TYPE_PRESETS: dict[str, tuple[str, list[int]]] = {
    "vqc_100": ("vqc", [1, 0, 0]),
    "vqc_111": ("vqc", [1, 1, 1]),
}


def _normalize_model_type(label: str) -> tuple[str, list[int] | None]:
    base, preset = MODEL_TYPE_PRESETS.get(label, (label, None))
    return base, preset.copy() if preset is not None else None


def build_args(config: dict[str, Any]) -> ExperimentArgs:
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
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
        log_wandb=logging_cfg.get("log_wandb", False),
        wandb_project=logging_cfg.get("wandb_project", "vqc_reproduction"),
        wandb_entity=logging_cfg.get("wandb_entity"),
        device=config.get("device", "cpu"),
    )

    requested_model = config.get("experiment", {}).get("model_type", "vqc")
    base_model, preset_state = _normalize_model_type(requested_model)
    if preset_state is not None:
        args.initial_state = preset_state
    args.requested_model_type = requested_model
    args.set_model_type(base_model)
    return args


def _visualize_datasets(
    datasets: dict[str, dict[str, torch.Tensor]], figures_dir: Path
) -> None:
    dataset_fig_dir = figures_dir / "datasets"
    dataset_fig_dir.mkdir(parents=True, exist_ok=True)
    for name in DATASET_ORDER:
        if name not in datasets:
            continue
        payload = datasets[name]
        x = torch.cat((payload["x_train"], payload["x_test"]))
        y = torch.cat((payload["y_train"], payload["y_test"]))
        plot_dataset_samples(
            x,
            y,
            f"{name.title()} dataset",
            dataset_fig_dir / f"{name}_scatter.png",
        )


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = cfg.get("data", {})
    datasets = prepare_datasets(data_cfg)
    if data_cfg.get("visualize", False):
        _visualize_datasets(datasets, figures_dir)

    args = build_args(cfg)
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

    outputs_cfg = cfg.get("outputs", {})
    if outputs_cfg.get("save_training_curves", True):
        plot_training_metrics(
            results,
            figures_dir / "training_metrics.png",
            dataset_order=DATASET_ORDER,
        )

    if outputs_cfg.get("save_decision_boundaries", True):
        boundary_dir = figures_dir / "decision_boundaries"
        boundary_dir.mkdir(parents=True, exist_ok=True)
        for entry in best_models:
            plot_decision_boundary(
                entry,
                boundary_dir / f"{entry['dataset']}_{args.requested_model_type}.png",
            )

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = ["train_and_evaluate", "build_args"]
