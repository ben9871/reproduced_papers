import pytest

from .common import load_cli_schema
from runtime_lib import build_cli_parser


def test_cli_help_exits_cleanly():
    parser, _ = build_cli_parser(load_cli_schema())
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
