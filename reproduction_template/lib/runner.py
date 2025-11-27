from __future__ import annotations

import json
import logging
import random
from pathlib import Path


def setup_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if getattr(torch.cuda, "is_available", lambda: False)():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - optional dependency
        pass


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Starting training")
    logger.debug("Resolved config: %s", json.dumps(cfg, indent=2))
    artifact = run_dir / "done.txt"
    artifact.write_text(
        "Training placeholder complete. Replace with actual implementation.\n",
        encoding="utf-8",
    )
    logger.info("Wrote placeholder artifact: %s", artifact)
