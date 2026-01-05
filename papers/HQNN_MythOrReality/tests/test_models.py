from __future__ import annotations

import common  # noqa: F401  (ensures project paths are on sys.path)
import pytest
from models.hqnn import enumerate_architectures


@pytest.mark.parametrize("feature_dim,num_classes", [(4, 2), (6, 3)])
def test_architectures_sorted_by_params(feature_dim: int, num_classes: int):
    specs = enumerate_architectures(feature_dim, num_classes)
    param_counts = [spec.param_count for spec in specs]
    assert param_counts == sorted(param_counts)
    assert len(specs) > 0
