#!/usr/bin/env python3
"""Repository-wide CLI entry point for reproduced papers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from runtime_lib import run_from_project

PAPERS_DIRNAME = "papers"
PROJECT_MARKERS = (
    Path("configs") / "defaults.json",
    Path("configs") / "cli.json",
    Path("lib") / "runner.py",
)


def _find_repo_root() -> Path:
    return Path(__file__).resolve().parent


def _is_project_dir(path: Path) -> bool:
    path = path.resolve()
    if not path.is_dir():
        return False
    return all((path / marker).exists() for marker in PROJECT_MARKERS)


def _iter_project_dirs(repo_root: Path) -> list[Path]:
    bases = [repo_root / PAPERS_DIRNAME, repo_root]
    seen: set[str] = set()
    found: list[Path] = []
    for base in bases:
        if not base.exists():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name in seen:
                continue
            if _is_project_dir(child):
                found.append(child)
                seen.add(name)
    return sorted(found, key=lambda p: p.name)


def _find_enclosing_project_dir(start: Path, repo_root: Path) -> Path | None:
    start = start.resolve()
    for path in [start, *start.parents]:
        if _is_project_dir(path):
            return path
        if path == repo_root:
            break
    return None


def _resolve_named_project(repo_root: Path, name: str) -> Path | None:
    raw = Path(name).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                repo_root / raw,
                repo_root / PAPERS_DIRNAME / raw,
                repo_root / raw.name,
                repo_root / PAPERS_DIRNAME / raw.name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _require_project_dir(
    parser: argparse.ArgumentParser, project_dir: Path | None
) -> Path:
    if project_dir is None:
        parser.error(
            "Cannot infer paper directory. Specify --paper/--paper-name/--paper-dir "
            "or run the command from inside a paper directory."
        )
    if not _is_project_dir(project_dir):
        parser.error(
            "Invalid paper directory (expected configs/defaults.json, configs/cli.json, "
            f"and lib/runner.py): {project_dir}"
        )
    return project_dir.resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--paper",
        "--paper-name",
        "--paper_name",
        "-p",
        dest="paper",
        help="Target paper folder under papers/ (e.g., QLSTM, QORC)",
    )
    parser.add_argument(
        "--paper-dir",
        "--paper_dir",
        dest="paper_dir",
        default=None,
        help="Explicit path to a paper directory (overrides --paper)",
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        dest="list_papers",
        help=(
            "List available paper folders that contain configs/defaults.json, "
            "configs/cli.json, and lib/runner.py"
        ),
    )
    parser.add_argument(
        "--project",
        "--project-name",
        "--project_name",
        dest="paper",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--project-dir",
        "--project_dir",
        dest="paper_dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--list-projects",
        action="store_true",
        dest="list_papers",
        help=argparse.SUPPRESS,
    )
    known, remaining = parser.parse_known_args(argv)

    repo_root = _find_repo_root()
    if known.list_papers:
        for path in _iter_project_dirs(repo_root):
            print(path.relative_to(repo_root))
        return 0

    project_dir: Path | None
    if known.paper:
        project_dir = _resolve_named_project(repo_root, known.paper)
    elif known.paper_dir:
        project_dir = Path(known.paper_dir).expanduser().resolve()
    else:
        project_dir = _find_enclosing_project_dir(Path.cwd(), repo_root)

    project_dir = _require_project_dir(parser, project_dir)
    run_from_project(project_dir, remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
