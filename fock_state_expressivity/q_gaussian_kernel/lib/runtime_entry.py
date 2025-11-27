"""Runtime entrypoints for the Quantum Gaussian kernel reproduction."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data.datasets import build_gaussian_grid, prepare_classification_data
from utils.plotting import (
    plot_accuracy_bars,
    plot_dataset_examples,
    plot_gaussian_fits,
)

from lib.training import (
    evaluate_classical_rbf,
    evaluate_quantum_classifiers,
    summarize_sampler,
    train_sampler,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


def _resolve_project_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoints(
    best_models: list[dict[str, Any]], dest_dir: Path
) -> list[dict[str, Any]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for entry in best_models:
        sigma_str = f"{entry['sigma_value']:.2f}"
        filename = f"sigma{sigma_str}_n{entry['photons']}.pth"
        path = dest_dir / filename
        torch.save(entry["state_dict"], path)
        manifest.append(
            {
                "sigma_label": entry["sigma_label"],
                "sigma_value": entry["sigma_value"],
                "photons": entry["photons"],
                "path": str(path),
            }
        )
    return manifest


def load_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    entries = data.get("entries", data)
    resolved: list[dict[str, Any]] = []
    for entry in entries:
        entry_path = Path(entry["path"])
        if not entry_path.is_absolute():
            entry_path = (path.parent / entry_path).resolve()
        resolved.append({**entry, "path": str(entry_path)})
    return resolved


def summarize_classification(
    quantum_entries: list[dict[str, Any]],
    classical_entries: list[dict[str, Any]],
) -> str:
    lines = ["Classification accuracy summary", ""]
    for entry in quantum_entries:
        lines.append(
            f"{entry['sigma_label']} | n={entry['n_photons']}: "
            f"circ={entry['circular_acc']:.3f}, moon={entry['moon_acc']:.3f}, blob={entry['blob_acc']:.3f}"
        )
    lines.append("")
    lines.append("Classical RBF baselines:")
    for entry in classical_entries:
        lines.append(
            f"{entry['sigma_label']}: "
            f"circ={entry['circular_acc']:.3f}, moon={entry['moon_acc']:.3f}, blob={entry['blob_acc']:.3f}"
        )
    return "\n".join(lines)


def _default_checkpoint_dir(cfg: dict[str, Any]) -> Path:
    sampler_cfg = cfg.get("sampler", {})
    custom = _resolve_project_path(sampler_cfg.get("checkpoint_dir"))
    if custom is not None:
        return custom
    return PROJECT_ROOT / "models"


def run_sampler_task(cfg: dict[str, Any], run_dir: Path, device: torch.device) -> None:
    sampler_cfg = cfg.get("sampler", {})
    training_cfg = dict(cfg.get("training", {}))
    if "num_runs" not in training_cfg:
        training_cfg["num_runs"] = 3

    grid = build_gaussian_grid(sampler_cfg.get("grid", {}))
    photon_counts = sampler_cfg.get("photons", [2, 4, 6, 8, 10])

    LOGGER.info(
        "Sampler settings -> photons: %s, sigma count: %s",
        photon_counts,
        len(grid.sigma_values),
    )

    results, best_models = train_sampler(
        cfg.get("model", {}),
        training_cfg,
        photon_counts,
        grid,
        device=device,
    )

    summary = summarize_sampler(results)
    (run_dir / "summary.txt").write_text(summary)
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    figures_dir = _ensure_dir(run_dir / "figures")
    if cfg.get("outputs", {}).get("save_learned_functions", True):
        plot_gaussian_fits(grid, best_models, figures_dir / "learned_vs_target.png")

    checkpoint_dir = _default_checkpoint_dir(cfg)
    manifest = save_checkpoints(best_models, checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": manifest}, indent=2))
    LOGGER.info("Saved sampler checkpoints to %s", manifest_path.resolve())

    export_dir = sampler_cfg.get("export_dir")
    if export_dir:
        export_path = _resolve_project_path(export_dir) or Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        export_entries: list[dict[str, Any]] = []
        for entry in manifest:
            src = Path(entry["path"]).resolve()
            dst = export_path / Path(entry["path"]).name
            shutil.copy2(src, dst)
            export_entries.append({**entry, "path": str(dst.resolve())})
        (export_path / "manifest.json").write_text(
            json.dumps({"entries": export_entries}, indent=2)
        )
        LOGGER.info("Copied sampler checkpoints to %s", export_path.resolve())


def run_classification_task(
    cfg: dict[str, Any], run_dir: Path, device: torch.device
) -> None:
    classification_cfg = cfg.get("classification", {})
    manifest_path = _resolve_project_path(classification_cfg.get("manifest"))
    if manifest_path is None:
        manifest_path = PROJECT_ROOT / "models" / "manifest.json"
    if not manifest_path.exists():
        LOGGER.warning(
            "Manifest '%s' not found. Run sampler task or provide --manifest.",
            manifest_path,
        )
        return

    LOGGER.info("Classification task -> using manifest: %s", manifest_path)
    manifest_entries = load_manifest(manifest_path)
    datasets = prepare_classification_data(classification_cfg.get("data", {}))
    LOGGER.info("Loaded datasets: %s", ", ".join(datasets.keys()))

    checkpoints = []
    for entry in manifest_entries:
        state_dict = torch.load(entry["path"], map_location="cpu")
        checkpoints.append(
            {
                "sigma_label": entry["sigma_label"],
                "sigma_value": entry["sigma_value"],
                "photons": entry["photons"],
                "state_dict": state_dict,
            }
        )

    figures_dir = _ensure_dir(run_dir / "figures")
    if cfg.get("outputs", {}).get("save_dataset_plots", True):
        dataset_fig = _ensure_dir(figures_dir / "datasets")
        plot_dataset_examples(datasets, dataset_fig / "classification_datasets.png")

    quantum_results = evaluate_quantum_classifiers(
        checkpoints,
        datasets,
        cfg.get("model", {}),
        device,
    )
    sigma_values = sorted({entry["sigma_value"] for entry in manifest_entries})
    classical_results = evaluate_classical_rbf(datasets, sigma_values)

    summary = summarize_classification(quantum_results, classical_results)
    (run_dir / "summary.txt").write_text(summary)
    (run_dir / "quantum_metrics.json").write_text(json.dumps(quantum_results, indent=2))
    (run_dir / "classical_metrics.json").write_text(
        json.dumps(classical_results, indent=2)
    )

    if cfg.get("outputs", {}).get("save_accuracy_bars", True):
        plot_accuracy_bars(
            quantum_results,
            classical_results,
            figures_dir / "svm_accuracy.png",
        )

    def _avg(entry: dict[str, float]) -> float:
        return float(
            (
                entry.get("circular_acc", 0.0)
                + entry.get("moon_acc", 0.0)
                + entry.get("blob_acc", 0.0)
            )
            / 3.0
        )

    if quantum_results:
        best_q = max(quantum_results, key=_avg)
        LOGGER.info(
            "Best quantum accuracy (avg): %.3f [%s, n=%s]",
            _avg(best_q),
            best_q["sigma_label"],
            best_q["n_photons"],
        )
    if classical_results:
        best_c = max(classical_results, key=_avg)
        LOGGER.info(
            "Best classical accuracy (avg): %.3f [%s]",
            _avg(best_c),
            best_c["sigma_label"],
        )


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    setup_seed(int(cfg.get("seed", 1337)))
    device = torch.device(cfg.get("device", "cpu"))

    task = cfg.get("experiment", {}).get("task", "sampler")
    LOGGER.info("Running task: %s", task.upper())

    if task == "sampler":
        run_sampler_task(cfg, run_dir, device)
    elif task == "classify":
        run_classification_task(cfg, run_dir, device)
    else:  # pragma: no cover - config validation
        raise ValueError(f"Unknown task: {task}")

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = [
    "setup_seed",
    "train_and_evaluate",
    "run_sampler_task",
    "run_classification_task",
]
