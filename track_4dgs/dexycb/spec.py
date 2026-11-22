"""One DexYCB capture."""

import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..blender.kernels import project_visibility, sample_near_surface, track_bind_points
from ..tracker import Query, Track
from ..utils import selected_frame_indices
from .camera import load_extrinsics, load_intrinsics
from .frame import DexYCBCameraSpec, DexYCBFrame
from .hand import (
    DEFAULT_MANO_LEFT,
    DEFAULT_MANO_LEFT_URL,
    DEFAULT_MANO_RIGHT,
    DEFAULT_MANO_RIGHT_URL,
    posed_mano_meshes,
)
from .object import posed_ycb_meshes
from .yaml import load_yaml


@dataclass(frozen=True)
class DexYCBSpec:
    """One DexYCB capture. Cameras and posed meshes are read in :meth:`load`.

    ``sequence_dir`` contains ``meta.yml`` and one image folder per camera.
    ``calibration_dir`` contains ``intrinsics/``, ``extrinsics_*/``, and ``mano_*/``.
    ``models_dir`` contains one folder per YCB object, each with ``textured_simple.obj``.
    ``mano_right`` and ``mano_left`` are the local MANO pickles. When either cannot
    be loaded, ``mano_right_url`` or ``mano_left_url`` is downloaded to that path.
    Tracks are ``num_points`` samples on the posed hand and YCB meshes.
    A hand or object missing from any selected frame is omitted. The load
    raises ``ValueError`` when every hand and every object is missing from
    at least one selected frame.
    Points are drawn at ``init_dataset_index`` and carried with that triangle.
    ``visible_cameras`` keeps a sample only when more cameras than this see it.
    ``inward_distance`` and ``outward_distance`` move a sample along the face normal.
    ``occlusion_eps`` is the occlusion tolerance in world units.
    Frames are ``range(frame_start, frame_end + 1, frame_step)`` using 0-based
    frame indices. ``frame_end`` defaults to the last frame. ``frame_step``
    defaults to 1. ``init_dataset_index`` indexes that range and is the query.
    """

    sequence_dir: str
    calibration_dir: str
    models_dir: str
    frame_start: int = 0
    frame_end: Optional[int] = None
    frame_step: int = 1
    init_dataset_index: int = 0
    load_depth: bool = False
    num_points: int = 256
    mano_right: str = DEFAULT_MANO_RIGHT
    mano_right_url: str = DEFAULT_MANO_RIGHT_URL
    mano_left: str = DEFAULT_MANO_LEFT
    mano_left_url: str = DEFAULT_MANO_LEFT_URL
    visible_cameras: int = 2
    inward_distance: float = 0.0
    outward_distance: float = 0.0
    occlusion_eps: float = 1e-2

    def __post_init__(self):
        if self.frame_step == 0:
            raise ValueError("frame_step must not be 0")
        if self.num_points < 1:
            raise ValueError("num_points must be positive")
        if self.inward_distance < 0 or self.outward_distance < 0:
            raise ValueError("inward_distance and outward_distance must be non-negative")
        if self.occlusion_eps < 0:
            raise ValueError("occlusion_eps must be non-negative")
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
        indices = selected_frame_indices(self.frame_start, self.frame_end, self.frame_step, int(meta["num_frames"]))
        hand_verts, hand_faces = posed_mano_meshes(
            self.sequence_dir, self.calibration_dir, meta, indices,
            self.mano_right, self.mano_right_url, self.mano_left, self.mano_left_url,
        )
        object_verts, object_faces = posed_ycb_meshes(
            self.sequence_dir, self.calibration_dir, self.models_dir, meta, indices,
        )
        verts = hand_verts + object_verts
        faces = hand_faces + object_faces
        if not verts:
            raise ValueError("no hand or YCB object has a pose on every selected frame")
        mesh_device = torch.device("cpu" if device is None else device)
        verts = [mesh.to(mesh_device) for mesh in verts]
        faces = [mesh.to(mesh_device) for mesh in faces]
        warp_device = str(mesh_device)
        with torch.no_grad():
            binding = sample_near_surface(
                [mesh[self.init_dataset_index] for mesh in verts],
                faces,
                frames[self.init_dataset_index],
                self.num_points,
                self.visible_cameras,
                self.inward_distance,
                self.outward_distance,
                self.occlusion_eps,
                warp_device,
            )
            frame_indices = torch.full(
                (binding.xyz.shape[0],), self.init_dataset_index, dtype=torch.long, device=binding.xyz.device,
            )
            query = Query.from_projection(binding.xyz, frames, frame_indices)
            worlds, valid, world_verts, scene_faces, _owner = track_bind_points(verts, faces, frames, binding)
            points, visibility = project_visibility(
                query, frames, worlds, world_verts, scene_faces, valid,
                self.occlusion_eps, warp_device, binding.face,
            )
            n_views, n_frames, n_points, _ = points.shape
            confidence = points.new_ones((n_views, n_frames, n_points))
            mask = valid.to(dtype=torch.bool).expand(n_frames, n_points)
            tracks = [
                Track(
                    points=points[view_index],
                    visibility=visibility[view_index],
                    confidence=confidence[view_index],
                    mask=mask,
                )
                for view_index in range(n_views)
            ]
        return query, frames, tracks
