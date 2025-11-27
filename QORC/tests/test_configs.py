from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime_lib.config import load_config

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"


def test_defaults_has_description_and_required_keys() -> None:
    cfg = load_config(CONFIG_DIR / "defaults.json")
    assert cfg["description"], "defaults.json must document the experiment"
    for key in ("xp_type", "outdir", "n_photons", "n_modes"):
        assert key in cfg, f"Missing '{key}' in defaults.json"


def test_cli_schema_matches_defaults_path() -> None:
    runtime_meta = json.loads((CONFIG_DIR / "runtime.json").read_text())
    defaults_path = runtime_meta["defaults_path"]
    cli_schema_path = runtime_meta["cli_schema_path"]
    runner_callable = runtime_meta["runner_callable"]

    assert (PROJECT_DIR / defaults_path).exists(), "runtime defaults_path missing"
    assert (PROJECT_DIR / cli_schema_path).exists(), "runtime cli schema missing"
    assert runner_callable.endswith("lib.runner.train_and_evaluate"), (
        "QORC runtime must call lib.runner.train_and_evaluate"
    )
