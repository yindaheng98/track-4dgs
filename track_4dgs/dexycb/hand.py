"""DexYCB posed MANO hand meshes."""

import os
import pickle
import urllib.request

import numpy as np
import torch

from .object import load_c2w
from .yaml import load_yaml

DEFAULT_MANO_RIGHT = os.path.join(os.path.dirname(__file__), "MANO_RIGHT.pkl")
DEFAULT_MANO_LEFT = os.path.join(os.path.dirname(__file__), "MANO_LEFT.pkl")
# minimal-hand/mano/models from https://github.com/ut-vision/SiMHand
DEFAULT_MANO_RIGHT_URL = "https://github.com/ut-vision/SiMHand/raw/main/minimal-hand/mano/models/MANO_RIGHT.pkl"
DEFAULT_MANO_LEFT_URL = "https://github.com/ut-vision/SiMHand/raw/main/minimal-hand/mano/models/MANO_LEFT.pkl"
# The pickle stores the root parent as 4294967295. The rest of this tree does not change.
MANO_PARENTS = (-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14)


def load_pose_m(sequence_dir: str, serials: list[str], indices: list[int]) -> torch.Tensor:
    """``pose_m`` for every camera on the selected frames.

    Returns ``(V, F, H, 51)`` float32. ``0:48`` are the MANO PCA pose
    coefficients and ``48:51`` is the translation, in meters, in that camera's
    frame. An all-zero pose is missing.
    """

    views = []
    count = None
    for serial in serials:
        folder = os.path.join(sequence_dir, serial)
        frames = []
        for frame_index in indices:
            path = os.path.join(folder, f"labels_{frame_index:06d}.npz")
            with np.load(path) as label:
                pose = np.asarray(label["pose_m"], dtype=np.float32)
            if pose.ndim != 2 or pose.shape[1] != 51:
                raise ValueError(f"{path} pose_m has shape {pose.shape}, expected (H, 51)")
            if count is None:
                count = pose.shape[0]
            elif pose.shape[0] != count:
                raise ValueError(f"{path} pose_m has {pose.shape[0]} hands, expected {count}")
            frames.append(pose)
        views.append(np.stack(frames))
    return torch.from_numpy(np.stack(views))


def load_mano(path: str, url: str) -> dict[str, torch.Tensor]:
    """Load one MANO pickle from ``path``.

    A missing file, or one that cannot be read as a MANO model, is replaced by
    downloading ``url`` into ``path``. The returned tensors are the ones
    :func:`mano_vertices` multiplies: template ``(778, 3)``, shape blend
    ``(778, 3, 10)``, pose blend ``(778, 3, 135)``, joint regressor ``(16, 778)``,
    skinning weights ``(778, 16)``, faces ``(1538, 3)``, PCA components ``(45, 45)``,
    and the mean finger pose ``(45,)``.
    """

    try:
        return read_mano_pickle(path)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, pickle.UnpicklingError):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        temporary = path + ".download"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(temporary, "wb") as handle:
                handle.write(response.read())
            os.replace(temporary, path)
        finally:
            if os.path.isfile(temporary):
                os.remove(temporary)
        return read_mano_pickle(path)


class ChumpyArray:
    """Stand-in for a chumpy array stored in an official MANO pickle.

    The pickle references ``chumpy`` classes. This object keeps the stored state
    so :func:`read_mano_pickle` can read ``shapedirs`` without chumpy installed.
    """

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.args = args
        return obj

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.x = state


class ManoUnpickler(pickle.Unpickler):
    """Unpickler that loads every ``chumpy`` class as :class:`ChumpyArray`."""

    def find_class(self, module, name):
        if module.startswith("chumpy"):
            return ChumpyArray
        return super().find_class(module, name)


def read_mano_pickle(path: str) -> dict[str, torch.Tensor]:
    """Read the official MANO pickle. The layout is fixed.

    778 vertices, 1538 faces, 16 joints, 45 PCA coefficients. ``shapedirs`` stores
    20 shape components and MANO uses the first 10. That array is a chumpy view,
    so :class:`ManoUnpickler` replaces the chumpy classes. ``J_regressor`` is a
    SciPy sparse matrix.
    """

    with open(path, "rb") as handle:
        raw = ManoUnpickler(handle, encoding="latin1").load()
    return {
        "v_template": torch.from_numpy(np.asarray(raw["v_template"], dtype=np.float32)),
        "shapedirs": torch.from_numpy(np.asarray(raw["shapedirs"].a.x, dtype=np.float32)[:, :, :10]),
        "posedirs": torch.from_numpy(np.asarray(raw["posedirs"], dtype=np.float32)),
        "j_regressor": torch.from_numpy(np.asarray(raw["J_regressor"].toarray(), dtype=np.float32)),
        "weights": torch.from_numpy(np.asarray(raw["weights"], dtype=np.float32)),
        "faces": torch.from_numpy(np.asarray(raw["f"], dtype=np.int32)),
        "components": torch.from_numpy(np.asarray(raw["hands_components"], dtype=np.float32)),
        "mean": torch.from_numpy(np.asarray(raw["hands_mean"], dtype=np.float32)),
    }


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Axis-angle ``(N, 3)`` to rotation matrices ``(N, 3, 3)``."""

    angle = torch.linalg.vector_norm(axis_angle + 1e-8, dim=1, keepdim=True)
    direction = axis_angle / angle
    cos = torch.cos(angle).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(-1)
    rx, ry, rz = direction[:, 0:1], direction[:, 1:2], direction[:, 2:3]
    zeros = torch.zeros_like(rx)
    skew = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros,
    ], dim=1).reshape(-1, 3, 3)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).unsqueeze(0)
    return identity + sin * skew + (1 - cos) * torch.matmul(skew, skew)


def skinning_transforms(rotations: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    """``(B, 16, 4, 4)`` transforms for :data:`MANO_PARENTS`. Identity rotations leave the rest vertices in place."""

    relative = joints.clone()
    for index, parent in enumerate(MANO_PARENTS):
        if parent >= 0:
            relative[:, index] = joints[:, index] - joints[:, parent]
    upper = torch.cat([rotations, relative.unsqueeze(-1)], dim=-1)
    lower = rotations.new_zeros(rotations.shape[0], 16, 1, 4)
    lower[..., 3] = 1
    local = torch.cat([upper, lower], dim=-2)
    chain = [local[:, 0]]
    for index in range(1, 16):
        chain.append(torch.matmul(chain[MANO_PARENTS[index]], local[:, index]))
    transforms = torch.stack(chain, dim=1)
    rest = torch.cat([joints, joints.new_zeros(joints.shape[0], 16, 1)], dim=-1).unsqueeze(-1)
    rotated = torch.matmul(transforms, rest)
    skin = transforms.clone()
    skin[:, :, :3, 3] = transforms[:, :, :3, 3] - rotated[:, :, :3, 0]
    return skin


def mano_vertices(model: dict[str, torch.Tensor], pose: torch.Tensor, betas: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    """Pose the template. ``pose`` is ``(B, 48)``, ``betas`` is ``(10,)``, ``trans`` is ``(B, 3)``.

    The first 3 pose values are the global axis-angle. The other 45 are PCA
    coefficients. The finger axis-angle is ``hands_mean + coefficients @ hands_components``.
    Returns ``(B, 778, 3)`` meters.
    """

    betas = betas.unsqueeze(0).expand(pose.shape[0], -1)
    finger = pose[:, 3:] @ model["components"] + model["mean"]
    full_pose = torch.cat([pose[:, :3], finger], dim=1)
    rotation = axis_angle_to_matrix(full_pose.reshape(-1, 3)).reshape(pose.shape[0], 16, 3, 3)
    v_shaped = model["v_template"] + torch.einsum("vcd,bd->bvc", model["shapedirs"], betas)
    joints = torch.einsum("jv,bvc->bjc", model["j_regressor"], v_shaped)
    identity = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
    pose_feature = (rotation[:, 1:] - identity).reshape(pose.shape[0], -1)
    v_posed = v_shaped + torch.einsum("vcp,bp->bvc", model["posedirs"], pose_feature)
    transforms = skinning_transforms(rotation, joints)
    skin = torch.matmul(model["weights"], transforms.reshape(pose.shape[0], 16, 16))
    skin = skin.reshape(pose.shape[0], 778, 4, 4)
    homo = torch.cat([v_posed, v_posed.new_ones(pose.shape[0], 778, 1)], dim=2)
    return torch.matmul(skin, homo.unsqueeze(-1))[:, :, :3, 0] + trans.unsqueeze(1)


def posed_mano_meshes(
        sequence_dir: str,
        calibration_dir: str,
        meta: dict,
        indices: list[int],
        mano_right: str = DEFAULT_MANO_RIGHT,
        mano_right_url: str = DEFAULT_MANO_RIGHT_URL,
        mano_left: str = DEFAULT_MANO_LEFT,
        mano_left_url: str = DEFAULT_MANO_LEFT_URL,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pose every hand that has a ``pose_m`` on every selected frame.

    ``meta`` is ``meta.yml``. ``indices`` are the selected frame numbers.
    Sides and calibration ids come from ``meta["mano_sides"]`` and
    ``meta["mano_calib"]``. Betas are ``calibration_dir/mano_<id>/mano.yml``.
    ``mano_right`` and ``mano_left`` are the local pickles. When either cannot
    be read, its URL is downloaded into that path.
    A frame uses the first camera whose pose is not all zeros, then that
    camera's extrinsics map the vertices into the capture world. The mapping
    is the one used by :func:`~track_4dgs.dexycb.object.posed_ycb_meshes`.
    Returns one ``(F, 778, 3)`` tensor and one ``(1538, 3)`` face tensor per hand.
    """

    files = {"right": (mano_right, mano_right_url), "left": (mano_left, mano_left_url)}
    if "mano_sides" not in meta or "mano_calib" not in meta:
        raise ValueError(f"{sequence_dir} has no mano_sides or mano_calib")
    sides = [str(side) for side in meta["mano_sides"]]
    calibs = list(meta["mano_calib"])
    if len(sides) != len(calibs) or len(sides) == 0:
        raise ValueError(f"{sequence_dir} mano_sides and mano_calib do not name the same hands")
    serials = [str(serial) for serial in meta["serials"]]
    poses = load_pose_m(sequence_dir, serials, indices)
    c2w = load_c2w(calibration_dir, meta)
    if poses.shape[2] != len(sides):
        raise ValueError(f"pose_m has {poses.shape[2]} hands and meta.yml lists {len(sides)}")
    missing = poses.abs().sum(dim=-1) == 0
    available = ~missing
    keep = available.any(dim=0).all(dim=0)
    if not bool(keep.any()):
        raise ValueError("no hand has a pose on every selected frame")
    view_index = available.long().argmax(dim=0)
    frame_index = torch.arange(poses.shape[1])
    models = {}
    betas = {}
    verts = []
    mesh_faces = []
    for hand in keep.nonzero(as_tuple=False).flatten().tolist():
        side = sides[hand]
        if side not in files:
            raise ValueError(f"unknown MANO side {side}")
        model = models.get(side)
        if model is None:
            model = load_mano(*files[side])
            models[side] = model
        calib = calibs[hand]
        shape = betas.get(calib)
        if shape is None:
            path = os.path.join(calibration_dir, f"mano_{calib}", "mano.yml")
            shape = torch.tensor(load_yaml(path)["betas"], dtype=torch.float32)
            betas[calib] = shape
        views = view_index[:, hand]
        pose = poses[views, frame_index, hand]
        verts_camera = mano_vertices(model, pose[:, :48], shape, pose[:, 48:51])
        hand_c2w = c2w[views]
        verts_world = torch.einsum("fij,fvj->fvi", hand_c2w[:, :3, :3], verts_camera) + hand_c2w[:, :3, 3].unsqueeze(1)
        verts.append(verts_world)
        mesh_faces.append(model["faces"])
    return verts, mesh_faces
