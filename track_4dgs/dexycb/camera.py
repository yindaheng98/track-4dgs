"""DexYCB camera calibration."""

import os

import numpy as np
import torch

from .frame import IMAGE_HEIGHT, IMAGE_WIDTH
from .yaml import load_yaml


def load_extrinsics(calibration_dir: str, sequence_meta: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """DexYCB stores a camera-to-world 3x4. Return world-to-camera ``R`` and ``T`` for each serial."""

    path = os.path.join(calibration_dir, f"extrinsics_{sequence_meta['extrinsics']}", "extrinsics.yml")
    tables = load_yaml(path)["extrinsics"]
    rotations = []
    translations = []
    for serial in sequence_meta["serials"]:
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3] = np.asarray(tables[str(serial)], dtype=np.float64).reshape(3, 4)
        w2c = np.linalg.inv(c2w)
        rotations.append(torch.tensor(w2c[:3, :3], dtype=torch.float32))
        translations.append(torch.tensor(w2c[:3, 3], dtype=torch.float32))
    return rotations, translations


def load_intrinsics(calibration_dir: str, sequence_meta: dict) -> tuple[list[float], list[float], list[float], list[float]]:
    focals_x = []
    focals_y = []
    centers_x = []
    centers_y = []
    for serial in sequence_meta["serials"]:
        path = os.path.join(calibration_dir, "intrinsics", f"{serial}_{IMAGE_WIDTH}x{IMAGE_HEIGHT}.yml")
        color = load_yaml(path)["color"]
        focals_x.append(float(color["fx"]))
        focals_y.append(float(color["fy"]))
        centers_x.append(float(color["ppx"]))
        centers_y.append(float(color["ppy"]))
    return focals_x, focals_y, centers_x, centers_y
