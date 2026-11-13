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
