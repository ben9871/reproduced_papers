from __future__ import annotations

import json
from pathlib import Path


def deep_update(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def default_config() -> dict:
    return {
        "seed": 0,
        "device": "cuda" if _has_cuda() else "cpu",
        "outdir": "outdir",
        "logging": {"level": "info"},
        "experiment": {
            "generator": "damped_shm",  # sin|damped_shm|logsine|ma_noise|csv
            "csv_path": None,
            "seq_length": 4,
            "train_split": 0.67,
            "fmt": "png",
            "snapshot_epochs": [],  # e.g., [1, 15, 30, 100]
            "plot_width": 6.0,
        },
        "model": {
            "type": "qlstm",  # qlstm|lstm|qlstm_photonic
            "hidden_size": 5,
            "vqc_depth": 4,
            "use_preencoders": False,
            # Photonic-specific defaults (ignored by other models)
            "shots": 0,
            "use_photonic_head": False,
        },
        "training": {
            "epochs": 50,
            "batch_size": 10,
            "lr": 1e-2,
        },
    }


def _has_cuda() -> bool:
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False
