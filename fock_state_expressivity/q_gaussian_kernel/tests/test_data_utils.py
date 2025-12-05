# ruff: noqa: E402
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from q_gaussian_kernel.data.datasets import (
    build_gaussian_grid,
    prepare_classification_data,
)


def test_gaussian_grid_targets_match_sigmas():
    cfg = {"step": 0.1, "span": 1.0, "sigmas": [1.0, 0.5]}
    grid = build_gaussian_grid(cfg)
    assert len(grid.targets) == 2
    assert grid.delta.shape == grid.targets[0].shape


def test_prepare_classification_data_shapes(tmp_path):
    cfg = {
        "cache_dir": tmp_path.as_posix(),
        "force_regenerate": True,
        "num_samples": 50,
        "test_size": 0.2,
        "datasets": {
            "circular": {"noise": 0.05},
            "moon": {"noise": 0.1},
            "blob": {"cluster_std": 2.0},
        },
    }
    data = prepare_classification_data(cfg)
    assert set(data.keys()) == {"circular", "moon", "blob"}
    for splits in data.values():
        assert splits["x_train"].shape[1] == 2
        assert splits["x_test"].shape[1] == 2
