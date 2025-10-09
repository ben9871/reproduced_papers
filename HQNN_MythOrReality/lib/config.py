from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """Load a JSON configuration file."""
    ext = path.suffix.lower()
    if ext != ".json":
        raise ValueError(f"Unsupported config extension: {ext}. Expected .json")
    return json.loads(path.read_text())


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base``."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def default_config() -> Dict[str, Any]:
    """Return the default configuration for HQNN spiral experiments."""
    return {
        "seed": 42,
        "outdir": "results",
        "device": "cpu",
        "dataset": {
            "name": "spiral",
            "num_instances": 1875,
            "num_classes": 3,
            "feature_grid": [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "test_size": 0.2,
            "batch_size": 64,
        },
        "model": {
            "accuracy_threshold": 90.0,
            "repetitions": 5,
            "binning": "linear",
            "embedding": "learned",
            "init": "none",
        },
        "training": {
            "epochs": 25,
            "lr": 0.05,
        },
        "logging": {"level": "info"},
        "results": {"filename": "HQNN_MM_bs{batch_size}_lr{lr}.json"},
    }
