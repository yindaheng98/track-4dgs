# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/core/checkpoint.py
"""Checkpoint save/load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# https://github.com/Lijiaxin0111/Open-d4rt/blob/403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3/src/core/checkpoint.py#L11-L18
def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)
