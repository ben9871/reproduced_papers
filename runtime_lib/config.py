from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON config file and ensure it contains a description."""
    logger = logging.getLogger(__name__)
    logger.info("Loading config file:%s", path)
    ext = path.suffix.lower()
    if ext != ".json":
        raise ValueError(
            f"Unsupported config extension (JSON required in reproduced_papers): {ext}"
        )

    with path.open("r", encoding="utf-8") as handler:
        data = json.load(handler)

    description = data.get("description")
    if description is None:
        raise ValueError("Config files must provide a 'description' field for traceability")
    logger.info(" JSON Description:%s", description)
    return data


def deep_update(
    base: Mapping[str, Any] | dict[str, Any],
    updates: Mapping[str, Any] | dict[str, Any],
) -> dict[str, Any]:
    """Recursively merge dictionaries, returning the mutated base copy."""

    if not isinstance(base, dict):
        raise TypeError("deep_update base must be a dict")
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)  # type: ignore[arg-type]
        else:
            base[key] = value  # type: ignore[index]
    return base
