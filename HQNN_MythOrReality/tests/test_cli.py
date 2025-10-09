from __future__ import annotations

import pathlib
import sys

import pytest
from common import load_implementation_module

_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))


def test_cli_help_exits_cleanly():
    impl = load_implementation_module()
    parser = impl.build_arg_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_resolve_config_overrides():
    impl = load_implementation_module()
    parser = impl.build_arg_parser()
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
        ]
    )
    cfg = impl.resolve_config(args)
    assert cfg["seed"] == 123
    assert cfg["dataset"]["batch_size"] == 16
    assert cfg["training"]["lr"] == pytest.approx(0.1)
    assert cfg["dataset"]["feature_grid"] == [4, 6]
    assert cfg["model"]["accuracy_threshold"] == pytest.approx(88.5)
