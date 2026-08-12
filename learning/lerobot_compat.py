"""Import LeRobot despite this repository's top-level ``datasets`` package.

LeRobot imports Hugging Face ``datasets`` while this project historically owns
the same top-level module name.  Limit the workaround to LeRobot construction:
temporarily expose the external package, import the ACT classes, then restore
the project's modules exactly as they were.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path
import sys


@lru_cache(maxsize=1)
def load_lerobot_act_classes():
    repository_root = Path(__file__).resolve().parents[1]
    local_datasets = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "datasets" or name.startswith("datasets.")
    }
    original_path = list(sys.path)
    try:
        for name in local_datasets:
            sys.modules.pop(name, None)
        sys.path = [
            entry
            for entry in sys.path
            if entry
            and Path(entry).resolve() != repository_root
        ]
        importlib.import_module("datasets")
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        return FeatureType, PolicyFeature, ACTConfig, ACTPolicy
    finally:
        for name in tuple(sys.modules):
            if name == "datasets" or name.startswith("datasets."):
                sys.modules.pop(name, None)
        sys.modules.update(local_datasets)
        sys.path = original_path
