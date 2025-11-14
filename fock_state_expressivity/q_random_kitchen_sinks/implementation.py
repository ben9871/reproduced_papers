from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from data.datasets import load_moons
from lib.training import run_rks_experiments
from utils.visualization import (
    plot_accuracy_heatmap,
    plot_combined_decisions,
    plot_dataset,
    save_kernel_heatmap,
)

DEFAULT_GAMMAS = list(range(1, 11))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum random kitchen sinks reproduction."
    )
    parser.add_argument(
        "--config", default="configs/defaults.json", help="Path to JSON config."
    )
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--outdir", help="Override output directory.")
    return parser.parse_args()


def load_config(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def prepare_run_directory(base: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def summarize_results(results, r_values, gamma_values) -> str:
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


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)

    seed = cli.seed or config.get("seed", 1337)
    set_seed(seed)

    outdir = Path(cli.outdir or config.get("outdir", "results"))
    run_dir = prepare_run_directory(outdir)
    fig_dir = run_dir / "figures"

    dataset_cfg = config.get("data", {})
    dataset = load_moons(dataset_cfg)
    plot_dataset(*dataset, fig_dir / "moon_dataset.png")

    results = run_rks_experiments(
        dataset,
        config.get("sweep", {}),
        config.get("model", {}),
        config.get("training", {}),
        config.get("classifier", {}),
        seed,
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

    r_values = config.get("sweep", {}).get("r_values", [1])
    gamma_values = config.get("sweep", {}).get("gamma_values", DEFAULT_GAMMAS)

    best_quantum = max(
        (entry for entry in results if entry["method"] == "quantum"),
        key=lambda e: e["accuracy"],
    )
    best_classical = max(
        (entry for entry in results if entry["method"] == "classical"),
        key=lambda e: e["accuracy"],
    )

    summary = summarize_results(
        results,
        r_values,
        gamma_values,
    )
    (run_dir / "summary.txt").write_text(summary)
    (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2))

    plot_accuracy_heatmap(
        results,
        r_values,
        gamma_values,
        "quantum",
        fig_dir / "accuracy_quantum.png",
    )
    plot_accuracy_heatmap(
        results,
        r_values,
        gamma_values,
        "classical",
        fig_dir / "accuracy_classical.png",
    )

    print(
        f"Best quantum accuracy: {best_quantum['accuracy']:.4f} (R={best_quantum['r']}, γ={best_quantum['gamma']})"
    )
    print(
        f"Best classical accuracy: {best_classical['accuracy']:.4f} (R={best_classical['r']}, γ={best_classical['gamma']})"
    )

    single_setting = len(r_values) == 1 and len(gamma_values) == 1
    if single_setting:
        save_kernel_heatmap(
            best_quantum["kernel_train"],
            dataset[2],
            f"Quantum kernel R={best_quantum['r']}, γ={best_quantum['gamma']}",
            fig_dir / "kernel_quantum.png",
        )
        save_kernel_heatmap(
            best_classical["kernel_train"],
            dataset[2],
            f"Classical kernel R={best_classical['r']}, γ={best_classical['gamma']}",
            fig_dir / "kernel_classical.png",
        )

    plot_combined_decisions(
        results,
        dataset,
        r_values,
        gamma_values,
        "quantum",
        fig_dir / "combined_decisions_quantum.png",
    )
    plot_combined_decisions(
        results,
        dataset,
        r_values,
        gamma_values,
        "classical",
        fig_dir / "combined_decisions_classical.png",
    )

    print(f"Artifacts saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
