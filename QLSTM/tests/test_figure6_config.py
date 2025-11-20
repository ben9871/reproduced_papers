import pathlib
import sys

# Ensure this tests directory is on sys.path to import shared helper
_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from common import _load_impl_module


def test_figure6_lstm_sine_smoke(tmp_path):
    impl = _load_impl_module()
    parser = impl.build_arg_parser()
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "configs" / "figure6_lstm_sine.json"
    assert cfg_path.exists(), "figure6_lstm_sine.json missing"
    # Force a very small epoch count for a quick smoke test
    args = parser.parse_args(["--config", str(cfg_path), "--epochs", "1"])  # override epochs to 1
    cfg = impl.resolve_config(args)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    impl.train_and_evaluate(cfg, run_dir)
    assert (run_dir / "model_final.pth").exists()
