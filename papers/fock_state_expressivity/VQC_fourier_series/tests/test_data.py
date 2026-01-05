# ruff: noqa: E402
import sys
from pathlib import Path

import torch

TEST_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VQC_fourier_series.lib.data import generate_dataset


def test_fourier_dataset_is_real():
    cfg = {
        "x_start": -1.0,
        "x_end": 1.0,
        "step": 0.5,
        "coefficients": [
            {"n": 0, "real": 0.1, "imag": 0.0},
            {"n": 1, "real": 0.2, "imag": 0.3},
        ],
    }

    dataset = generate_dataset(cfg)
    x = dataset["x"]
    y = dataset["y"]

    assert x.shape[0] == y.shape[0]
    assert not torch.is_complex(y)
