"""Posed YCB meshes for a DexYCB capture."""

import os

import numpy as np
import torch

from .camera import load_extrinsics

YCB_MODEL_NAMES = {
    1: "002_master_chef_can",
    2: "003_cracker_box",
    3: "004_sugar_box",
    4: "005_tomato_soup_can",
    5: "006_mustard_bottle",
    6: "007_tuna_fish_can",
    7: "008_pudding_box",
    8: "009_gelatin_box",
    9: "010_potted_meat_can",
    10: "011_banana",
    11: "019_pitcher_base",
    12: "021_bleach_cleanser",
    13: "024_bowl",
    14: "025_mug",
    15: "035_power_drill",
    16: "036_wood_block",
    17: "037_scissors",
    18: "040_large_marker",
    19: "051_large_clamp",
    20: "052_extra_large_clamp",
    21: "061_foam_brick",
}


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


def load_c2w(calibration_dir: str, sequence_meta: dict) -> torch.Tensor:
    """``(V, 4, 4)`` camera-to-world for each serial, in :func:`~track_4dgs.dexycb.camera.load_extrinsics` order."""

    rotations, translations = load_extrinsics(calibration_dir, sequence_meta)
    poses = []
    for rotation, translation in zip(rotations, translations, strict=True):
        w2c = torch.eye(4, dtype=rotation.dtype)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = translation
        poses.append(torch.linalg.inv(w2c))
    return torch.stack(poses)


def posed_ycb_meshes(
        sequence_dir: str,
        calibration_dir: str,
        models_dir: str,
        meta: dict,
        indices: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pose every YCB object that has a ``pose_y`` on every selected frame.

    ``meta`` is ``meta.yml``. ``indices`` are the selected frame numbers.
    Object ids come from ``meta["ycb_ids"]``. Poses are read from each camera's
    labels and mapped into the capture world with that camera's extrinsics.
    A frame uses the first camera whose pose is not all zeros.
    Meshes are ``models_dir/<name>/textured_simple.obj``.
    Returns one ``(F, V_i, 3)`` tensor and one ``(T_i, 3)`` face tensor per object.
    No such object returns two empty lists.
    """

    if "ycb_ids" not in meta:
        raise ValueError(f"{sequence_dir} has no ycb_ids")
    ycb_ids = [int(ycb_id) for ycb_id in meta["ycb_ids"]]
    serials = [str(serial) for serial in meta["serials"]]
    poses = load_pose_y(sequence_dir, serials, indices)
    c2w = load_c2w(calibration_dir, meta)
    if poses.shape[2] != len(ycb_ids):
        raise ValueError(f"pose_y has {poses.shape[2]} objects and meta.yml lists {len(ycb_ids)}")
    missing = poses.reshape(*poses.shape[:-2], -1).abs().sum(dim=-1) == 0
    available = ~missing
    keep = available.any(dim=0).all(dim=0)
    view_index = available.long().argmax(dim=0)
    frame_index = torch.arange(poses.shape[1])
    verts = []
    mesh_faces = []
    for obj in keep.nonzero(as_tuple=False).flatten().tolist():
        name = YCB_MODEL_NAMES.get(ycb_ids[obj])
        if name is None:
            raise ValueError(f"unknown YCB id {ycb_ids[obj]}")
        path = os.path.join(models_dir, name, "textured_simple.obj")
        vertices, faces = load_obj(path)
        views = view_index[:, obj]
        pose = poses[views, frame_index, obj]
        object_c2w = c2w[views]
        verts_camera = torch.einsum("fij,vj->fvi", pose[:, :3, :3], vertices) + pose[:, :3, 3].unsqueeze(1)
        verts_world = torch.einsum("fij,fvj->fvi", object_c2w[:, :3, :3], verts_camera) + object_c2w[:, :3, 3].unsqueeze(1)
        verts.append(verts_world)
        mesh_faces.append(faces)
    return verts, mesh_faces
