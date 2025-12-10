"""Utility helpers for QRKD reproduction."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_tasks(tasks) -> list[str]:
    if isinstance(tasks, str):
        return [t.strip() for t in tasks.split(",") if t.strip()]
    if isinstance(tasks, Iterable):
        return [str(t) for t in tasks]
    return []
