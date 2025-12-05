import json
from pathlib import Path

from lib.runner import main, set_seed


def test_run_example_config(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "example.json"
    assert config_path.exists(), "Missing configs/example.json"

    cfg = json.loads(config_path.read_text())
    cfg["outdir"] = str(tmp_path)

    set_seed(cfg.get("seed", 42))
    run_dir = Path(main(cfg))

    assert run_dir.exists()
    assert (run_dir / "summary_results.json").exists()
    assert (run_dir / "config_snapshot.json").exists()
    assert (run_dir / "done.txt").exists()
