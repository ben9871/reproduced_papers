#!/usr/bin/env python3
"""CLI entry point for reproducing HQNN spiral experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from data.data import SpiralDatasetConfig, load_spiral_dataset
from lib.config import deep_update, default_config, load_config
from models.hqnn import ArchitectureSpec, build_hqnn_model, enumerate_architectures
from torch.utils.data import DataLoader, TensorDataset
from utils.io import save_experiment_results
from utils.training import count_parameters, train_model


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(level: str = "info", log_file: Path | None = None) -> None:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    log_level = level_map.get(str(level).lower(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HQNN spiral reproduction experiments")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to JSON config file"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--outdir", type=str, default=None, help="Base output directory"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device string (cpu|cuda|mps)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Training epochs override"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size override"
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override")
    parser.add_argument(
        "--feature-grid",
        type=str,
        default=None,
        help="Comma-separated list of feature counts to evaluate",
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=None,
        help="Stop searching architectures once this validation accuracy is reached",
    )
    return parser


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = default_config()
    if args.config:
        path = Path(args.config)
        file_cfg = load_config(path)
        cfg = deep_update(cfg, file_cfg)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["dataset"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr
    if args.feature_grid:
        cfg["dataset"]["feature_grid"] = [
            int(x.strip()) for x in args.feature_grid.split(",") if x.strip()
        ]
    if args.accuracy_threshold is not None:
        cfg["model"]["accuracy_threshold"] = args.accuracy_threshold

    return cfg


def _spiral_config(
    base_cfg: dict[str, Any], feature_dim: int, seed: int
) -> SpiralDatasetConfig:
    return SpiralDatasetConfig(
        num_instances=base_cfg["num_instances"],
        num_features=feature_dim,
        num_classes=base_cfg["num_classes"],
        test_size=base_cfg.get("test_size", 0.2),
        random_state=seed,
    )


def _prepare_dataloaders(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader


def _format_results_filename(pattern: str, batch_size: int, lr: float) -> str:
    try:
        return pattern.format(batch_size=batch_size, lr=lr)
    except KeyError:
        return pattern


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    logger = logging.getLogger(__name__)

    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    results_cfg = cfg.get("results", {})

    device = torch.device(cfg["device"])
    batch_size = dataset_cfg["batch_size"]
    lr = training_cfg["lr"]

    results_filename = _format_results_filename(
        results_cfg.get("filename", "results.json"),
        batch_size=batch_size,
        lr=lr,
    )
    results_path = run_dir / results_filename

    logger.info("Writing aggregated results to %s", results_path)

    for feature_dim in dataset_cfg["feature_grid"]:
        logger.info("Evaluating architectures for %s features", feature_dim)
        architectures: list[ArchitectureSpec] = enumerate_architectures(
            feature_dim,
            dataset_cfg["num_classes"],
        )

        for spec in architectures:
            logger.debug(
                "Trying architecture modes=%s photons=%s no_bunching=%s params≈%s",
                spec.modes,
                spec.photons,
                spec.no_bunching,
                spec.param_count,
            )

            all_accs: list[float] = []
            last_curves: dict[str, list[float]] | None = None
            last_model: torch.nn.Module | None = None
            last_input_state: list[int] | None = None

            for repetition in range(model_cfg["repetitions"]):
                spiral_cfg = _spiral_config(
                    dataset_cfg,
                    feature_dim,
                    seed=cfg["seed"] + repetition,
                )
                x_train, x_val, y_train, y_val, _, _ = load_spiral_dataset(spiral_cfg)
                train_loader, val_loader = _prepare_dataloaders(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    batch_size=batch_size,
                )

                model, input_state, _ = build_hqnn_model(
                    feature_dim,
                    dataset_cfg["num_classes"],
                    spec,
                    device,
                )

                (
                    train_losses,
                    val_losses,
                    best_val_acc,
                    train_accs,
                    val_accs,
                ) = train_model(
                    model,
                    train_loader,
                    val_loader,
                    num_epochs=training_cfg["epochs"],
                    lr=training_cfg["lr"],
                    device=device,
                )
                all_accs.append(best_val_acc)
                last_curves = {
                    "train_loss": train_losses,
                    "val_loss": val_losses,
                    "train_acc": train_accs,
                    "val_acc": val_accs,
                }
                last_model = model
                last_input_state = input_state

            mean_acc = float(np.mean(all_accs))
            std_acc = float(np.std(all_accs))
            logger.info(
                "Architecture modes=%s photons=%s no_bunching=%s mean_acc=%.2f ± %.2f",
                spec.modes,
                spec.photons,
                spec.no_bunching,
                mean_acc,
                std_acc,
            )

            if last_model is None or last_curves is None or last_input_state is None:
                logger.warning("Skipping result logging due to missing state.")
                continue

            results = {
                "dataset": dataset_cfg.get("name", "spiral"),
                "lr": lr,
                "bs": batch_size,
                "nb_samples": dataset_cfg["num_instances"],
                "nb_features": feature_dim,
                "nb_classes": dataset_cfg["num_classes"],
                "modes": spec.modes,
                "nb_photons": spec.photons,
                "no_bunching": spec.no_bunching,
                "input_state": last_input_state,
                "binning": model_cfg.get("binning"),
                "embedding": model_cfg.get("embedding"),
                "init": model_cfg.get("init"),
                "BEST q ACC": mean_acc,
                "BEST q ACC std": std_acc,
                "q parameters": count_parameters(last_model),
                "q curves": last_curves,
            }

            save_experiment_results(results, results_path)

            if mean_acc >= model_cfg["accuracy_threshold"]:
                logger.info(
                    "Threshold %.2f reached (mean=%.2f). Moving to next feature size.",
                    model_cfg["accuracy_threshold"],
                    mean_acc,
                )
                break


def main(argv: list[str] | None = None) -> int:
    configure_logging("info")
    script_dir = Path(__file__).resolve().parent
    if Path.cwd().resolve() != script_dir:
        logging.info("Switching working directory to %s", script_dir)
        os.chdir(script_dir)

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = resolve_config(args)
    setup_seed(cfg["seed"])

    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = Path(cfg["outdir"])
    run_dir = base_out / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(cfg.get("logging", {}).get("level", "info"), run_dir / "run.log")
    (run_dir / "config_snapshot.json").write_text(json.dumps(cfg, indent=2))

    train_and_evaluate(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
