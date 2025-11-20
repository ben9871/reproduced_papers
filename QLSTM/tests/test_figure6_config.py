import pathlib

from .common import _load_impl_module


def test_figure6_lstm_sine_smoke(tmp_path):
    impl = _load_impl_module()
    parser = impl.build_arg_parser()
    cfg_path = (
        pathlib.Path(__file__).resolve().parents[1] / "configs" / "sine_lstm.json"
    )
    assert cfg_path.exists(), "sine_lstm.json missing"
    # Force a very small epoch count for a quick smoke test
    args = parser.parse_args(
        ["--config", str(cfg_path), "--epochs", "1"]
    )  # override epochs to 1
    cfg = impl.resolve_config(args)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    impl.train_and_evaluate(cfg, run_dir)
    assert (run_dir / "model_final.pth").exists()
