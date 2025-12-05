"""Nearest centroid QML reproduced paper library package.

This exposes the components used by the shared runner and notebooks.
"""

from .classifier import MLQuantumNearestCentroid, QuantumNearestCentroid

__all__ = ["QuantumNearestCentroid", "MLQuantumNearestCentroid"]

from .synthetic_data import generate_paper_datasets, generate_synthetic_data

__all__ = [
    "QuantumNearestCentroid",
    "MLQuantumNearestCentroid",
    "generate_synthetic_data",
    "generate_paper_datasets",
]
