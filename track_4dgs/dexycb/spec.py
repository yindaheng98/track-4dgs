"""One DexYCB capture and the readers that build its cameras and joints."""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import yaml

from ..tracker import Query, Track
from ..utils import selected_frame_indices
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


def load_joint_tracks(sequence_dir: str, sequence_meta: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """MANO ``joint_2d`` for every view.

    DexYCB writes ``-1, -1`` when that joint has no annotation. A joint that
    projects outside the image keeps its pixel coordinate. Bounds are applied
    later by :func:`~track_4dgs.dexycb.trajectory.query_and_tracks_from_trajectories`.
    """

    serials = [str(serial) for serial in sequence_meta["serials"]]
    num_frames = int(sequence_meta["num_frames"])
    views = []
    for serial in serials:
        folder = os.path.join(sequence_dir, serial)
        frames = []
        for frame_index in range(num_frames):
            with np.load(os.path.join(folder, f"labels_{frame_index:06d}.npz")) as label:
                joints = np.asarray(label["joint_2d"], dtype=np.float32)
            if joints.shape != (1, 21, 2):
                raise ValueError(
                    f"{folder} frame {frame_index} joint_2d has shape {joints.shape}, expected (1, 21, 2)"
                )
            frames.append(joints[0])
        views.append(np.stack(frames))
    points = torch.from_numpy(np.stack(views))
    missing = (points == -1).all(dim=-1)
    valid = torch.isfinite(points).all(dim=-1) & ~missing
    return points, valid


@dataclass(frozen=True)
class DexYCBSpec:
    """One DexYCB capture. Cameras and joints are read in :meth:`load`.

    ``sequence_dir`` contains ``meta.yml`` and one image folder per camera.
    ``calibration_dir`` contains ``intrinsics/`` and ``extrinsics_*/``.
    Frames are ``range(frame_start, frame_end + 1, frame_step)`` using 0-based
    frame indices. ``frame_end`` defaults to the last frame. ``frame_step``
    defaults to 1. ``init_dataset_index`` indexes that range and is the query.
    """

    sequence_dir: str
    calibration_dir: str
    frame_start: int = 0
    frame_end: Optional[int] = None
    frame_step: int = 1
    init_dataset_index: int = 0
    load_depth: bool = False

    def __post_init__(self):
        if self.frame_step == 0:
            raise ValueError("frame_step must not be 0")
        if self.init_dataset_index < 0:
            raise ValueError("--init-dataset-index must index one frame")
        if self.frame_end is not None and self.frame_step is not None:
            n_frames = len(range(self.frame_start, self.frame_end + 1, self.frame_step))
            if self.init_dataset_index >= n_frames:
                raise ValueError("--init-dataset-index must index one frame")

    def load(self, device) -> tuple[Query, list[DexYCBFrame], list[Track]]:
        meta = load_yaml(os.path.join(self.sequence_dir, "meta.yml"))
        serials = [str(serial) for serial in meta["serials"]]
        num_frames = int(meta["num_frames"])
        if len(serials) == 0:
            raise ValueError(f"{self.sequence_dir} has no cameras")
        if num_frames <= 0:
            raise ValueError(f"{self.sequence_dir} has no frames")
        indices = selected_frame_indices(self.frame_start, self.frame_end, self.frame_step, num_frames)
        if self.init_dataset_index >= len(indices):
            raise ValueError("--init-dataset-index must index one frame")
        rotations, translations = load_extrinsics(self.calibration_dir, meta)
        focals_x, focals_y, centers_x, centers_y = load_intrinsics(self.calibration_dir, meta)
        camera_specs = [[] for _ in indices]
        for serial, fx, fy, cx, cy, rotation, translation in zip(
                serials, focals_x, focals_y, centers_x, centers_y, rotations, translations, strict=True):
            folder = os.path.join(self.sequence_dir, serial)
            for slot, frame_index in enumerate(indices):
                camera_specs[slot].append(DexYCBCameraSpec(
                    serial=serial,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    rotation=rotation,
                    translation=translation,
                    image_path=os.path.join(folder, f"color_{frame_index:06d}.jpg"),
                    depth_path=os.path.join(folder, f"aligned_depth_to_color_{frame_index:06d}.png"),
                ))
        frames = [DexYCBFrame(specs, load_depth=self.load_depth, device=device) for specs in camera_specs]
        points, valid = load_joint_tracks(self.sequence_dir, meta)
        points = points[:, indices]
        valid = valid[:, indices]
        if device is not None:
            points = points.to(device)
            valid = valid.to(device)
        start_indices = torch.full(
            (points.shape[0], points.shape[2]),
            self.init_dataset_index,
            dtype=torch.int64,
            device=points.device,
        )
        query, tracks = query_and_tracks_from_trajectories(
            points,
            valid,
            [IMAGE_WIDTH] * len(serials),
            [IMAGE_HEIGHT] * len(serials),
            start_indices,
        )
        return query, frames, tracks
