import importlib.util
import pathlib


def _load_impl_module():
    impl_path = pathlib.Path(__file__).resolve().parents[1] / "implementation.py"
    assert impl_path.exists(), "implementation.py missing"
    spec = importlib.util.spec_from_file_location("impl", impl_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_help_exits_cleanly():
    impl = _load_impl_module()
    parser = impl.build_arg_parser()
    try:
        parser.parse_args(["--help"])  # argparse triggers SystemExit on --help
    except SystemExit as e:
        assert e.code == 0
    else:
        assert False, "Expected SystemExit when parsing --help"


def test_train_and_evaluate_writes_artifact(tmp_path):
    impl = _load_impl_module()
    parser = impl.build_arg_parser()
    args = parser.parse_args([])
    cfg = impl.resolve_config(args)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    impl.train_and_evaluate(cfg, run_dir)

    assert (run_dir / "done.txt").exists(), "Expected artifact file to be created"
