"""One DexYCB capture."""

import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..tracker import Query, Track
from ..utils import selected_frame_indices
from .camera import load_extrinsics, load_intrinsics
from .frame import IMAGE_HEIGHT, IMAGE_WIDTH, DexYCBCameraSpec, DexYCBFrame
from .hand import load_joint_tracks
from .trajectory import query_and_tracks_from_trajectories
from .yaml import load_yaml


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

    def load_frames(self, device) -> tuple[list[DexYCBFrame], dict]:
        """Read the selected cameras and ``meta.yml``.

        Frames are ``range(frame_start, frame_end + 1, frame_step)``.
        """

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
        return frames, meta

    def load(self, device) -> tuple[Query, list[DexYCBFrame], list[Track]]:
        frames, meta = self.load_frames(device)
        serials = [str(serial) for serial in meta["serials"]]
        num_frames = int(meta["num_frames"])
        indices = selected_frame_indices(self.frame_start, self.frame_end, self.frame_step, num_frames)
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
