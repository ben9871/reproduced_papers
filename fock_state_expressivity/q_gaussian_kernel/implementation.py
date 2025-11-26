from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data.datasets import build_gaussian_grid, prepare_classification_data
from lib.training import (
    evaluate_classical_rbf,
    evaluate_quantum_classifiers,
    summarize_sampler,
    train_sampler,
)
from utils.plotting import plot_accuracy_bars, plot_dataset_examples, plot_gaussian_fits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantum Gaussian kernel reproduction."
    )
    parser.add_argument(
        "--config", default="configs/defaults.json", help="Path to config JSON."
    )
    parser.add_argument(
        "--task", choices=["sampler", "classify"], help="Override experiment task."
    )
    parser.add_argument("--seed", type=int, help="Random seed override.")
    parser.add_argument("--outdir", help="Output directory base.")
    parser.add_argument("--device", help="Torch device string.")
    parser.add_argument("--manifest", help="Checkpoint manifest for classification.")
    parser.add_argument(
        "--checkpoint-dir", help="Directory containing saved kernel checkpoints."
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Override number of runs per photon count during sampling.",
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


def save_checkpoints(best_models: list[dict], dest_dir: Path) -> list[dict]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
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


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    entries = data.get("entries", data)
    for entry in entries:
        entry["path"] = (
            str((path.parent / entry["path"]).resolve())
            if not Path(entry["path"]).is_absolute()
            else entry["path"]
        )
    return entries


def summarize_classification(
    quantum_entries: list[dict], classical_entries: list[dict]
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


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    config.setdefault("experiment", {})

    seed = cli.seed or config.get("seed", 1337)
    set_seed(int(seed))

    outdir = Path(cli.outdir or config.get("outdir", "results"))
    run_dir = prepare_run_directory(outdir)
    device = torch.device(cli.device or config.get("device", "cpu"))

    task = cli.task or config["experiment"].get("task", "sampler")
    print(f"Running task: {task.upper()}")

    if task == "sampler":
        sampler_cfg = config.get("sampler", {})
        grid = build_gaussian_grid(sampler_cfg.get("grid", {}))
        photon_counts = sampler_cfg.get("photons", [2, 4, 6, 8, 10])
        training_cfg = dict(config.get("training", {}))
        if cli.num_runs is not None:
            training_cfg["num_runs"] = int(cli.num_runs)
            print(f"Overriding sampler num_runs -> {training_cfg['num_runs']}")
        elif "num_runs" not in training_cfg:
            training_cfg["num_runs"] = 3

        print(
            f"Sampler settings -> photons: {photon_counts}, σ count: {len(grid.sigma_values)}, "
            f"runs per combo: {training_cfg.get('num_runs', 'n/a')}"
        )
        results, best_models = train_sampler(
            config.get("model", {}),
            training_cfg,
            photon_counts,
            grid,
            device=device,
        )

        summary = summarize_sampler(results)
        (run_dir / "summary.txt").write_text(summary)
        (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))

        fig_dir = run_dir / "figures"
        if config.get("outputs", {}).get("save_learned_functions", True):
            plot_gaussian_fits(grid, best_models, fig_dir / "learned_vs_target.png")

        models_dir = Path("models")
        manifest = save_checkpoints(best_models, models_dir)
        manifest_path = models_dir / "manifest.json"
        manifest_path.write_text(json.dumps({"entries": manifest}, indent=2))
        print(f"Saved sampler checkpoints to {manifest_path.resolve()}")

        export_dir = sampler_cfg.get("export_dir")
        if export_dir:
            export_path = Path(export_dir)
            export_path.mkdir(parents=True, exist_ok=True)
            for entry in manifest:
                src = Path(entry["path"])
                dst = export_path / src.name
                shutil.copy2(src, dst)
                entry["path"] = str(dst.resolve())
            (export_path / "manifest.json").write_text(
                json.dumps({"entries": manifest}, indent=2)
            )

    elif task == "classify":
        classification_cfg = config.get("classification", {})
        manifest_path = (
            Path(cli.manifest)
            if cli.manifest
            else Path(classification_cfg.get("manifest", "models/manifest.json"))
        )
        if not manifest_path.exists():
            print(
                f"[Warning] Manifest '{manifest_path}' not found. "
                "Run 'python implementation.py --task sampler' first or provide --manifest."
            )
            return
        print(f"Classification task -> using manifest: {manifest_path}")
        manifest_entries = load_manifest(manifest_path)
        datasets = prepare_classification_data(classification_cfg.get("data", {}))
        print(f"Loaded datasets: {', '.join(datasets.keys())}")

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

        if config.get("outputs", {}).get("save_dataset_plots", True):
            plot_dataset_examples(
                datasets, run_dir / "figures" / "classification_datasets.png"
            )

        quantum_results = evaluate_quantum_classifiers(
            checkpoints, datasets, config.get("model", {}), device
        )
        sigma_values = sorted({entry["sigma_value"] for entry in manifest_entries})
        classical_results = evaluate_classical_rbf(datasets, sigma_values)

        summary = summarize_classification(quantum_results, classical_results)
        (run_dir / "summary.txt").write_text(summary)

        (run_dir / "quantum_metrics.json").write_text(
            json.dumps(quantum_results, indent=2)
        )
        (run_dir / "classical_metrics.json").write_text(
            json.dumps(classical_results, indent=2)
        )

        if config.get("outputs", {}).get("save_accuracy_bars", True):
            plot_accuracy_bars(
                quantum_results,
                classical_results,
                run_dir / "figures" / "svm_accuracy.png",
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
            print(
                f"Best quantum accuracy (avg over datasets): {_avg(best_q):.3f} "
                f"[{best_q['sigma_label']}, n={best_q['n_photons']}]"
            )
        if classical_results:
            best_c = max(classical_results, key=_avg)
            print(
                f"Best classical accuracy (avg over datasets): {_avg(best_c):.3f} "
                f"[{best_c['sigma_label']}]"
            )
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Artifacts saved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
