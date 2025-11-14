# ruff: noqa: E402
import sys
from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from q_random_kitchen_sinks.data.datasets import load_moons, target_function


def test_load_moons_shapes(tmp_path):
    cfg = {
        "n_samples": 50,
        "noise": 0.1,
        "random_state": 0,
        "cache_dir": tmp_path.as_posix(),
    }
    x_train, x_test, y_train, y_test = load_moons(cfg)
    assert x_train.shape[1] == 2
    assert x_test.shape[1] == 2
    assert len(y_train) + len(y_test) == 50


def test_target_function_dimension():
    x = np.linspace(0, 1, 5).reshape(1, -1)
    y = target_function(x)
    assert y.shape == x.shape
