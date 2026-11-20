"""One DexYCB capture."""

import os
from dataclasses import dataclass
from typing import Optional

import torch

from ..blender.kernels import project_visibility, sample_near_surface, track_bind_points
from ..tracker import Query, Track
from ..utils import selected_frame_indices
from .camera import load_extrinsics, load_intrinsics
from .frame import IMAGE_HEIGHT, IMAGE_WIDTH, DexYCBCameraSpec, DexYCBFrame
from .hand import load_joint_tracks
from .object import posed_ycb_meshes
from .trajectory import query_and_tracks_from_trajectories
from .yaml import load_yaml


@dataclass(frozen=True)
class DexYCBSpec:
    """One DexYCB capture. Cameras, joints, and posed meshes are read in :meth:`load`.

    ``sequence_dir`` contains ``meta.yml`` and one image folder per camera.
    ``calibration_dir`` contains ``intrinsics/`` and ``extrinsics_*/``.
    ``models_dir`` contains one folder per YCB object, each with ``textured_simple.obj``.
    Tracks are the 21 MANO joints, then ``num_points`` samples on the posed meshes.
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
        verts, faces = posed_ycb_meshes(
            self.sequence_dir, self.calibration_dir, self.models_dir, meta, indices,
        )
        mesh_device = torch.device("cpu" if device is None else device)
        verts = [obj.to(mesh_device) for obj in verts]
        faces = [obj.to(mesh_device) for obj in faces]
        warp_device = str(mesh_device)
        with torch.no_grad():
            binding = sample_near_surface(
                [obj[self.init_dataset_index] for obj in verts],
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
            mesh_query = Query.from_projection(binding.xyz, frames, frame_indices)
            worlds, mesh_valid, world_verts, scene_faces, _owner = track_bind_points(verts, faces, frames, binding)
            mesh_points, visibility = project_visibility(
                mesh_query, frames, worlds, world_verts, scene_faces, mesh_valid,
                self.occlusion_eps, warp_device, binding.face,
            )
            n_views, n_frames, n_mesh_points, _ = mesh_points.shape
            confidence = mesh_points.new_ones((n_views, n_frames, n_mesh_points))
            mask = mesh_valid.to(dtype=torch.bool).expand(n_frames, n_mesh_points)
            mesh_tracks = [
                Track(
                    points=mesh_points[view_index],
                    visibility=visibility[view_index],
                    confidence=confidence[view_index],
                    mask=mask,
                )
                for view_index in range(n_views)
            ]
        query = Query(
            points=torch.cat([query.points, mesh_query.points], dim=1),
            frame_indices=torch.cat([query.frame_indices, mesh_query.frame_indices], dim=1),
            in_image=torch.cat([query.in_image, mesh_query.in_image], dim=1),
        )
        tracks = [
            Track(
                points=torch.cat([hand.points, mesh.points], dim=1),
                visibility=torch.cat([hand.visibility, mesh.visibility], dim=1),
                confidence=torch.cat([hand.confidence, mesh.confidence], dim=1),
                mask=torch.cat([hand.mask, mesh.mask], dim=1),
            )
            for hand, mesh in zip(tracks, mesh_tracks, strict=True)
        ]
        return query, frames, tracks
