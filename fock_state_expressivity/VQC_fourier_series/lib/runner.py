"""Runtime entrypoints for the VQC Fourier-series expressivity reproduction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from data.fourier_series import generate_dataset
from utils.plotting import plot_learned_functions, plot_training_curves

from lib.training import summarize_results, train_models_multiple_runs
from lib.vqc import VQCFactory

LOGGER = logging.getLogger(__name__)


def _serialize_metrics(results: dict[str, Any]) -> dict[str, Any]:
    return {
        label: {
            "final_mses": [run["train_mses"][-1] for run in data["runs"]],
            "loss_curves": data["runs"],
            "initial_state": data["initial_state"],
        }
        for label, data in results.items()
    }


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    device = torch.device(cfg.get("device", "cpu"))

    dataset = generate_dataset(cfg.get("data", {}))
    x_tensor = dataset["x"]
    y_tensor = dataset["y"]

    model_factory = VQCFactory(cfg.get("model", {}))
    training_cfg = cfg.get("training", {})
    initial_states = training_cfg.get("initial_states", [])
    plotting_cfg = cfg.get("plotting", {})
    color_palette = plotting_cfg.get("colors")

    results, best_models = train_models_multiple_runs(
        model_factory,
        initial_states,
        x_tensor,
        y_tensor,
        training_cfg,
        device=device,
        colors=color_palette,
    )

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_text = summarize_results(results)
    (run_dir / "summary.txt").write_text(summary_text)
    (run_dir / "metrics.json").write_text(
        json.dumps(_serialize_metrics(results), indent=2)
    )

    logging_cfg = cfg.get("logging", {})
    show_plots = bool(plotting_cfg.get("show", False))

    if logging_cfg.get("save_training_curves", True):
        plot_training_curves(
            results,
            save_path=figures_dir / "training_curves.png",
            show=show_plots,
        )

    if logging_cfg.get("save_learned_functions", True):
        plot_learned_functions(
            best_models,
            x_tensor,
            y_tensor,
            save_path=figures_dir / "learned_vs_target.png",
            show=show_plots,
        )

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = ["train_and_evaluate"]
