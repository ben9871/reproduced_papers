"""Experiment input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_experiment_results(results: dict[str, Any], output_file: Path) -> int:
    """Append experiment results to a JSON lines file and return stored run count."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        try:
            existing: list[dict[str, Any]] = json.loads(output_file.read_text())
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []

    existing.append(results)
    output_file.write_text(json.dumps(existing, indent=4))
    return len(existing)
