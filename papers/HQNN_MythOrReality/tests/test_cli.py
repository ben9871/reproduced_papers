from __future__ import annotations

import common  # noqa: F401  (ensures project paths are on sys.path)
import pytest
from common import build_project_parser, load_defaults_copy, project_root

from runtime_lib.cli import apply_cli_overrides


def test_cli_help_exits_cleanly():
    parser, _ = build_project_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_resolve_config_overrides():
    parser, arg_defs = build_project_parser()
    args = parser.parse_args(
        [
            "--seed",
            "123",
            "--batch-size",
            "16",
            "--lr",
            "0.1",
            "--feature-grid",
            "4,6",
            "--accuracy-threshold",
            "88.5",
            "--figure",
        ]
    )
    cfg = load_defaults_copy()
    project_dir = project_root()
    cfg = apply_cli_overrides(cfg, args, arg_defs, project_dir, project_dir)
    assert cfg["seed"] == 123
    assert cfg["dataset"]["batch_size"] == 16
    assert cfg["training"]["lr"] == pytest.approx(0.1)
    assert cfg["dataset"]["feature_grid"] == [4, 6]
    assert cfg["model"]["accuracy_threshold"] == pytest.approx(88.5)
    assert cfg["results"]["make_threshold_figure"] is True
