import pathlib
import sys

# Ensure this tests directory is on sys.path to import shared helper
_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from common import _load_impl_module


def test_cli_help_exits_cleanly():
    impl = _load_impl_module()
    parser = impl.build_arg_parser()
    try:
        parser.parse_args(["--help"])  # argparse triggers SystemExit on --help
    except SystemExit as e:
        assert e.code == 0
    else:
        raise AssertionError("Expected SystemExit when parsing --help")

