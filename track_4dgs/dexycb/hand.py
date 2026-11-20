"""DexYCB MANO joint tracks."""

import os

import numpy as np
import torch

from .object import load_c2w
from .yaml import load_yaml

MANO_VERTEX_COUNT = 778
MANO_JOINT_COUNT = 16
MANO_BETA_COUNT = 10
MANO_PCA_COMPONENTS = 45
DEFAULT_MANO_RIGHT = os.path.join(os.path.dirname(__file__), "MANO_RIGHT.pkl")
DEFAULT_MANO_LEFT = os.path.join(os.path.dirname(__file__), "MANO_LEFT.pkl")
# minimal-hand/mano/models from https://github.com/ut-vision/SiMHand
DEFAULT_MANO_RIGHT_URL = "https://github.com/ut-vision/SiMHand/raw/main/minimal-hand/mano/models/MANO_RIGHT.pkl"
DEFAULT_MANO_LEFT_URL = "https://github.com/ut-vision/SiMHand/raw/main/minimal-hand/mano/models/MANO_LEFT.pkl"


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
