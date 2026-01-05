from __future__ import annotations

import json
from pathlib import Path


def _defaults_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "defaults.json"


def default_config() -> dict[str, object]:
    with _defaults_path().open("r", encoding="utf-8") as handle:
        return json.load(handle)
