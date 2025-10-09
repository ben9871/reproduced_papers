from __future__ import annotations

import pathlib
import sys

import pytest

_TESTS_DIR = pathlib.Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from common import load_implementation_module


@pytest.mark.parametrize("feature_dim,num_classes", [(4, 2), (6, 3)])
def test_architectures_sorted_by_params(feature_dim: int, num_classes: int):
    _ = load_implementation_module()
    from models.hqnn import enumerate_architectures

    specs = enumerate_architectures(feature_dim, num_classes)
    param_counts = [spec.param_count for spec in specs]
    assert param_counts == sorted(param_counts)
    assert len(specs) > 0
