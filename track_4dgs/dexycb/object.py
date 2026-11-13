"""DexYCB supervision from posed YCB meshes."""

import os
from argparse import Namespace
from dataclasses import dataclass, field

import numpy as np
import torch

from ..blender.kernels import project_visibility, sample_near_surface, track_bind_points
from ..tracker import Query, Track
from ..utils import selected_frame_indices
from .dataset import DexYCBDataset
from .frame import DexYCBFrame
from .spec import DexYCBSpec, load_extrinsics


def load_obj(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangles from a Wavefront ``.obj``.

    Vertex lines are ``v x y z``. Face indices are 1-based and may carry
    ``vertex/texcoord/normal`` suffixes. Polygons are fanned into triangles.
    Returns ``(V, 3)`` float vertices and ``(T, 3)`` int32 faces.
    """

    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                corners = [int(part.split("/")[0]) for part in line.split()[1:]]
                for index in range(1, len(corners) - 1):
                    faces.append((corners[0] - 1, corners[index] - 1, corners[index + 1] - 1))
    if not vertices or not faces:
        raise ValueError(f"{path} has no triangles")
    return torch.tensor(vertices, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32)


def load_pose_y(sequence_dir: str, serials: list[str], indices: list[int]) -> torch.Tensor:
    """``pose_y`` for every camera on the selected frames.

    Returns ``(V, F, O, 3, 4)`` float32. Each ``(3, 4)`` is ``[R|t]`` in that
    camera's frame. An all-zero pose is missing.
    """

    views = []
    count = None
    for serial in serials:
        folder = os.path.join(sequence_dir, serial)
        frames = []
        for frame_index in indices:
            path = os.path.join(folder, f"labels_{frame_index:06d}.npz")
            with np.load(path) as label:
                pose = np.asarray(label["pose_y"], dtype=np.float32)
            if pose.ndim != 3 or pose.shape[1:] != (3, 4):
                raise ValueError(f"{path} pose_y has shape {pose.shape}, expected (O, 3, 4)")
            if count is None:
                count = pose.shape[0]
            elif pose.shape[0] != count:
                raise ValueError(f"{path} pose_y has {pose.shape[0]} objects, expected {count}")
            frames.append(pose)
        views.append(np.stack(frames))
    return torch.from_numpy(np.stack(views))


class DexYCBObjectDataset(DexYCBDataset):
    """One sample is one DexYCB capture, tracked on its YCB meshes.

    ``--sequence`` groups match :class:`~track_4dgs.dexycb.dataset.DexYCBDataset`.
    ``--models-dir`` is the ``models/`` folder from ``models.tar.gz``.
    ``--num-points`` samples are drawn on the posed meshes at ``--init-dataset-index``
    and kept when more cameras than ``--visible-cameras`` see them.
    ``--inward-distance`` and ``--outward-distance`` move a sample along the face normal.
    ``--occlusion-eps`` is the occlusion tolerance in world units.
    """

    @staticmethod
    def add_common_argument(parser):
        """Register the calibration, mesh, and sampling flags and return ``parser``."""

        DexYCBDataset.add_common_argument(parser)
        parser.add_argument("--models-dir", required=True, type=str, help="Folder of YCB meshes. Each object has textured_simple.obj.")
        parser.add_argument("--num-points", default=256, type=int, help="Surface samples on the posed meshes. Defaults to 256.")
        parser.add_argument("--visible-cameras", default=0, type=int, help="Keep a point only when more cameras than this see it. Defaults to 0.")
        parser.add_argument("--inward-distance", default=0.0, type=float, help="How far points may move into the mesh along the face normal, in world units.")
        parser.add_argument("--outward-distance", default=0.0, type=float, help="How far points may move out of the mesh along the face normal, in world units.")
        parser.add_argument("--occlusion-eps", default=1e-2, type=float, help="Occlusion depth tolerance in world units. Defaults to 1e-2.")
        return parser

    @staticmethod
    def spec_from_args(common_args: Namespace, group_args: Namespace) -> DexYCBObjectSpec:
        """One :class:`DexYCBObjectSpec` for this ``--sequence`` group."""

        return DexYCBObjectSpec(
            sequence_dir=group_args.sequence,
            calibration_dir=common_args.calibration_dir,
            models_dir=common_args.models_dir,
            frame_start=group_args.frame_start,
            frame_end=group_args.frame_end,
            frame_step=group_args.frame_step,
            init_dataset_index=group_args.init_dataset_index,
            load_depth=common_args.load_depth,
            num_points=common_args.num_points,
            visible_cameras=common_args.visible_cameras,
            inward_distance=common_args.inward_distance,
            outward_distance=common_args.outward_distance,
            occlusion_eps=common_args.occlusion_eps,
        )
