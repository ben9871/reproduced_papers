from __future__ import annotations

import importlib
from typing import Callable


def import_callable(path_spec: str) -> Callable[..., object]:
    module_name, attr = path_spec.rsplit(".", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):  # pragma: no cover - defensive
        raise TypeError(f"Callable required: {path_spec}")
    return fn
