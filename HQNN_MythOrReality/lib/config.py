from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from runtime_lib.config import deep_update as _deep_update
from runtime_lib.config import load_config as _load_config

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULTS_PATH = _PROJECT_ROOT / "configs" / "defaults.json"


def load_config(path: Path) -> dict[str, Any]:
    """Proxy to the shared runtime config loader."""
    return _load_config(path)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Expose the shared deep merge helper for backward compatibility."""
    return _deep_update(base, updates)


def default_config() -> dict[str, Any]:
    """Read the JSON defaults so legacy helpers stay in sync with runtime."""
    return copy.deepcopy(load_config(_DEFAULTS_PATH))
