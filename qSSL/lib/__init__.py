from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from .defaults import default_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

from runtime_lib import config as _shared_config

module_name = __name__ + ".config"
config_module = ModuleType(module_name)
config_module.load_config = _shared_config.load_config
config_module.deep_update = _shared_config.deep_update
config_module.default_config = default_config
sys.modules[module_name] = config_module
config = config_module

__all__ = ["config", "default_config"]
