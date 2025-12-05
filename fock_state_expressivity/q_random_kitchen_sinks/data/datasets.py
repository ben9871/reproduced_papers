from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_moons(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate, split, and scale the moons dataset."""
    n_samples = int(cfg.get("n_samples", 200))
    noise = float(cfg.get("noise", 0.2))
    random_state = int(cfg.get("random_state", 42))
    test_prop = float(cfg.get("test_prop", 0.4))
    scaling = cfg.get("scaling", "Standard")

    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_prop, random_state=random_state
    )

    if scaling == "Standard":
        scaler = StandardScaler()
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaling}")

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    cache_dir = Path(cfg.get("cache_dir", "data/cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "moons_x_train.npy", x_train)
    np.save(cache_dir / "moons_x_test.npy", x_test)
    np.save(cache_dir / "moons_y_train.npy", y_train)
    np.save(cache_dir / "moons_y_test.npy", y_test)

    return x_train, x_test, y_train, y_test


def target_function(x):
    """Cosine-based target used for random feature fitting."""
    return np.sqrt(2) * np.cos(x)
