"""Nearest centroid QML reproduced paper library package.

This exposes the components used by the shared runner and notebooks.
"""

from .classifier import QuantumNearestCentroid, MLQuantumNearestCentroid

__all__ = ["QuantumNearestCentroid", "MLQuantumNearestCentroid"]

from .synthetic_data import generate_synthetic_data, generate_paper_datasets

__all__ = [
    "QuantumNearestCentroid",
    "MLQuantumNearestCentroid", 
    "generate_synthetic_data",
    "generate_paper_datasets",
]
