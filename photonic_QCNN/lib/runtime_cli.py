from __future__ import annotations

from typing import Any

from .runtime_entry import DATASET_CHOICES


def _apply_to_runs(cfg: dict[str, Any], key: str, value: int) -> dict[str, Any]:
    runs = cfg.setdefault("runs", {})
    for dataset in DATASET_CHOICES:
        runs.setdefault(dataset, {})[key] = value
    return cfg


def apply_batch_size(cfg: dict[str, Any], value: Any, _arg_def: dict[str, Any]) -> dict[str, Any]:
    return _apply_to_runs(cfg, "batch_size", int(value))


def apply_n_runs(cfg: dict[str, Any], value: Any, _arg_def: dict[str, Any]) -> dict[str, Any]:
    return _apply_to_runs(cfg, "n_runs", int(value))
