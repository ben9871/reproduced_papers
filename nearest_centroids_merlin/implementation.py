#!/usr/bin/env python
"""
Local convenience CLI for the nearest_centroids reproduction.

This is *not* the MerLin shared runner. In the full MerLin repo, you typically run:

    python implementation.py --project nearest_centroids --config nearest_centroids/configs/example.json

from the *repository root*.

This file just lets you do:

    python implementation.py --config configs/example.json

from inside the nearest_centroids folder, which is handy for local dev.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

# This is the MerLin entrypoint your configs/runtime.json should point to:
#     "runner_callable": "lib.runner.main"
from lib.runner import main as run_experiment


def _here() -> Path:
    """Return the project directory (folder containing this file)."""
    return Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nearest Centroids paper reproduction (local runner)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/example.json",
        help=(
            "Path to JSON config, relative to this folder. "
            "Defaults to configs/example.json"
        ),
    )
    return parser


def load_config(path: Path) -> Dict[str, Any]:
    """Load a JSON config file into a dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    project_root = _here()

    # Resolve config path relative to this file if not absolute
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    cfg = load_config(cfg_path)

    # Delegate to the MerLin-style runner
    run_experiment(cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
