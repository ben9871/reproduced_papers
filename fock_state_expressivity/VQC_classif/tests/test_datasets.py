# ruff: noqa: E402
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VQC_classif.data.datasets import DATASET_ORDER, prepare_datasets


def test_prepare_datasets_creates_all_splits(tmp_path, monkeypatch):
    cfg = {
        "cache_dir": tmp_path.as_posix(),
        "force_regenerate": True,
        "order": list(DATASET_ORDER),
        "datasets": {
            "linear": {"num_samples": 50, "class_sep": 1.2},
            "circular": {"num_samples": 50, "noise": 0.1},
            "moon": {"num_samples": 50, "noise": 0.2},
        },
        "test_size": 0.2,
        "standardize": True,
        "seed": 1,
    }
    datasets = prepare_datasets(cfg)
    assert set(datasets.keys()) == set(DATASET_ORDER)
    for _name, payload in datasets.items():
        assert "x_train" in payload and "x_test" in payload
        assert payload["x_train"].shape[1] == 2
