#!/usr/bin/env python3
"""Repository-wide CLI entry point for reproduced papers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime_lib import run_from_project


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--project",
        "-p",
        help="Target project folder (e.g., QLSTM, QORC)",
    )
    parser.add_argument(
        "--project-dir",
        default=None,
        help="Explicit path to a project directory (overrides --project)",
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        help="List available project folders that contain configs/runtime.json",
    )
    known, remaining = parser.parse_known_args(argv)

    repo_root = _find_repo_root()
    if known.list_projects:
        for path in sorted(repo_root.iterdir()):
            if not path.is_dir():
                continue
            runtime_file = path / "configs" / "runtime.json"
            if runtime_file.exists():
                rel = path.relative_to(repo_root)
                print(rel)
        return 0

    if known.project:
        project_dir = (repo_root / known.project).resolve()
    elif known.project_dir:
        project_dir = Path(known.project_dir).resolve()
    else:
        parser.error("Specify --project or --project-dir")

    if not project_dir.exists():
        parser.error(f"Project directory does not exist: {project_dir}")

    run_from_project(project_dir, remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
