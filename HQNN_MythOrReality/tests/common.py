"""Shared helpers for HQNN test suite."""

from __future__ import annotations

import importlib
import pathlib
import sys


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _ensure_path() -> None:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def load_implementation_module():
    """Import and return the ``implementation`` module."""
    _ensure_path()
    return importlib.import_module("implementation")
