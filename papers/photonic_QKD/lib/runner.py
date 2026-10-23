from __future__ import annotations

import logging
from pathlib import Path


def train_and_evaluate(cfg, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    artifact = run_dir / "done.txt"
    artifact.write_text(
        "photonic_QKD scaffold run complete; PQKD training not yet implemented.\n",
        encoding="utf-8",
    )
    logger.info("Wrote scaffold artifact: %s", artifact)
