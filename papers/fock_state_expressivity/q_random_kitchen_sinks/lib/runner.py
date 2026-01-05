"""Runtime entrypoints for the Quantum random kitchen sinks reproduction."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data import load_moons
from utils.visualization import (
    plot_accuracy_heatmap,
    plot_combined_decisions,
    plot_dataset,
    save_kernel_heatmap,
)

from lib.training import run_rks_experiments

DEFAULT_GAMMAS = list(range(1, 11))
LOGGER = logging.getLogger(__name__)


def summarize_results(
    results: list[dict[str, Any]],
    r_values: Iterable[int],
    gamma_values: Iterable[int],
) -> str:
    lines = ["Quantum vs Classical Random Kitchen Sinks", ""]
    for method in ["quantum", "classical"]:
        lines.append(method.capitalize())
        for r in r_values:
            for gamma in gamma_values:
                subset = [
                    entry
                    for entry in results
                    if entry["method"] == method
                    and entry["r"] == r
                    and entry["gamma"] == gamma
                ]
                if subset:
                    avg = sum(entry["accuracy"] for entry in subset) / len(subset)
                    lines.append(f"  R={r:>3}, gamma={gamma:>2}: {avg:.4f}")
        lines.append("")
    return "\n".join(lines)


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.get("data", {})
    dataset = load_moons(dataset_cfg)
    plot_dataset(*dataset, figures_dir / "moon_dataset.png")

    results = run_rks_experiments(
        dataset,
        cfg.get("sweep", {}),
        cfg.get("model", {}),
        cfg.get("training", {}),
        cfg.get("classifier", {}),
        int(cfg.get("seed", 1337)),
    )

    metrics = [
        {
            "method": entry["method"],
            "r": entry["r"],
            "gamma": entry["gamma"],
            "repeat": entry["repeat"],
            "accuracy": entry["accuracy"],
        }
        for entry in results
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    r_values = cfg.get("sweep", {}).get("r_values", [1])
    gamma_values = cfg.get("sweep", {}).get("gamma_values", DEFAULT_GAMMAS)

    summary = summarize_results(results, r_values, gamma_values)
    (run_dir / "summary.txt").write_text(summary)

    plot_accuracy_heatmap(
        results,
        r_values,
        gamma_values,
        "quantum",
        figures_dir / "accuracy_quantum.png",
    )
    plot_accuracy_heatmap(
        results,
        r_values,
        gamma_values,
        "classical",
        figures_dir / "accuracy_classical.png",
    )

    best_quantum = max(
        (entry for entry in results if entry["method"] == "quantum"),
        key=lambda e: e["accuracy"],
    )
    best_classical = max(
        (entry for entry in results if entry["method"] == "classical"),
        key=lambda e: e["accuracy"],
    )

    LOGGER.info(
        "Best quantum accuracy %.4f (R=%s, gamma=%s)",
        best_quantum["accuracy"],
        best_quantum["r"],
        best_quantum["gamma"],
    )
    LOGGER.info(
        "Best classical accuracy %.4f (R=%s, gamma=%s)",
        best_classical["accuracy"],
        best_classical["r"],
        best_classical["gamma"],
    )

    if len(r_values) == 1 and len(gamma_values) == 1:
        _, _, y_train, _ = dataset
        save_kernel_heatmap(
            best_quantum["kernel_train"],
            y_train,
            f"Quantum kernel R={best_quantum['r']}, γ={best_quantum['gamma']}",
            figures_dir / "kernel_quantum.png",
        )
        save_kernel_heatmap(
            best_classical["kernel_train"],
            y_train,
            f"Classical kernel R={best_classical['r']}, γ={best_classical['gamma']}",
            figures_dir / "kernel_classical.png",
        )

    plot_combined_decisions(
        results,
        dataset,
        r_values,
        gamma_values,
        "quantum",
        figures_dir / "combined_decisions_quantum.png",
    )
    plot_combined_decisions(
        results,
        dataset,
        r_values,
        gamma_values,
        "classical",
        figures_dir / "combined_decisions_classical.png",
    )

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = ["train_and_evaluate", "summarize_results"]
