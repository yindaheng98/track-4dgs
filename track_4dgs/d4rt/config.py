# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/core/config.py
"""Configuration helpers with recursive node access and override support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/core/config.py#L71-L76
def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data
