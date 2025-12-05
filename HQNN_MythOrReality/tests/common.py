"""Shared helpers for the HQNN test suite."""

from __future__ import annotations

import copy
import json
import pathlib
import sys
from typing import Any


def _prepare_paths() -> tuple[pathlib.Path, pathlib.Path]:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    for candidate in (project_root, repo_root):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return project_root, repo_root


_PROJECT_ROOT, _REPO_ROOT = _prepare_paths()

from runtime_lib.cli import build_cli_parser  # noqa: E402
from runtime_lib.config import load_config  # noqa: E402
from runtime_lib.runtime import load_runtime_meta  # noqa: E402


def project_root() -> pathlib.Path:
    return _PROJECT_ROOT


def repo_root() -> pathlib.Path:
    return _REPO_ROOT


def load_runtime_metadata() -> dict[str, Any]:
    return load_runtime_meta(_PROJECT_ROOT)


def build_project_parser() -> tuple[Any, list[dict[str, Any]]]:
    meta = load_runtime_metadata()
    cli_schema_path = project_root() / meta["cli_schema_path"]
    cli_schema = json.loads(cli_schema_path.read_text())
    global_cli = json.loads(
        (repo_root() / "runtime_lib" / "global_cli.json").read_text()
    )
    cli_schema.setdefault("arguments", [])
    cli_schema["arguments"].extend(global_cli.get("arguments", []))
    parser, arg_defs = build_cli_parser(cli_schema)
    return parser, arg_defs


def load_defaults_copy() -> dict[str, Any]:
    meta = load_runtime_metadata()
    defaults = load_config(project_root() / meta["defaults_path"])
    return copy.deepcopy(defaults)
