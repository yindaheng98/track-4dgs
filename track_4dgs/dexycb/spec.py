"""One DexYCB capture and the readers that build its cameras and joints."""

import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from ..tracker import Query, Track
from .frame import IMAGE_HEIGHT, IMAGE_WIDTH, DexYCBCameraSpec, DexYCBFrame
from .trajectory import query_and_tracks_from_trajectories


class DexYAMLLoader(yaml.SafeLoader):
    pass


def construct_tuple(loader, node):
    return loader.construct_sequence(node)


DexYAMLLoader.add_constructor("tag:yaml.org,2002:python/tuple", construct_tuple)


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=DexYAMLLoader)


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
