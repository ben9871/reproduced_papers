from pathlib import Path

from lib.config import deep_update, load_config
from lib.runner import train_and_evaluate

from runtime_lib.seed import seed_everything


def test_figure6_lstm_sine_smoke(tmp_path):
    project_dir = Path(__file__).resolve().parents[1]
    defaults = load_config(project_dir / "configs" / "defaults.json")
    cfg_path = project_dir / "configs" / "sine_lstm.json"
    assert cfg_path.exists(), "sine_lstm.json missing"
    cfg = deep_update(defaults, load_config(cfg_path))
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = max(2, cfg["training"].get("batch_size", 2))
    seed_everything(cfg["seed"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_and_evaluate(cfg, run_dir)
    assert (run_dir / "model_final.pth").exists()
