"""Utilities for seeding common scientific Python stacks consistently."""

from __future__ import annotations

import logging
import os
import random
from typing import Final

LOGGER: Final = logging.getLogger("runtime.seed")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs when available.

    Parameters
    ----------
    seed:
        The integer seed applied everywhere.
    deterministic:
        When True, enforces deterministic CuDNN kernels (if PyTorch is available).
    """

    LOGGER.info("Seeding global RNGs with seed=%s", seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.debug("NumPy seeding skipped: %s", exc)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if getattr(torch.cuda, "is_available", lambda: False)():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.debug("PyTorch seeding skipped: %s", exc)


__all__ = ["seed_everything"]
