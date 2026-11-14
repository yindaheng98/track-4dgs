"""Surface samples on the posed SMPL meshes stored with a Harmony4D sequence."""

import os
import urllib.request

import numpy as np
import torch

from ..blender.kernels import project_visibility, sample_near_surface, track_bind_points
from ..tracker import Query

SMPL_VERTEX_COUNT = 6890
SMPL_WRISTS = (20, 21)
HAND_RADIUS = 0.08
DEFAULT_SMPL_FACES = os.path.join(os.path.dirname(__file__), "smpl_faces.npy")
# src/tf_smpl/smpl_faces.npy from https://github.com/akanazawa/hmr
DEFAULT_SMPL_FACES_URL = "https://github.com/akanazawa/hmr/raw/master/src/tf_smpl/smpl_faces.npy"


def load_smpl_faces(path: str, url: str) -> torch.Tensor:
    """Load the ``(13776, 3)`` SMPL triangles from ``path``.

    Each row indexes three of the 6890 vertices. A missing file, or one that
    is not that array, is replaced by downloading ``url`` into ``path``.
    ``DEFAULT_SMPL_FACES_URL`` is ``src/tf_smpl/smpl_faces.npy`` from
    https://github.com/akanazawa/hmr.
    """

    try:
        faces = np.load(path)
        if faces.shape != (13776, 3):
            raise ValueError(f"{path} has shape {faces.shape}, expected (13776, 3)")
    except (OSError, ValueError):
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
        faces = np.load(path)
        if faces.shape != (13776, 3):
            raise ValueError(f"{path} has shape {faces.shape}, expected (13776, 3)")
    return torch.tensor(np.ascontiguousarray(faces), dtype=torch.int32)


def body_face_mask(vertices: torch.Tensor, joints: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Body triangles: every corner is more than 8 cm from either SMPL wrist.

    Joints 20 and 21 are the wrists. Vertices inside that radius are the mean
    hand, so their triangles are not sampled. ``True`` means the triangle may
    be drawn.
    """

    hand = torch.cdist(vertices, joints[list(SMPL_WRISTS)]).amin(dim=1) < HAND_RADIUS
    return ~hand[faces.long()].any(dim=1)


def load_body_surface(
        sequence: str,
        frame_indices: list[int],
        frames,
        init_dataset_index: int,
        num_points: int,
        transform: torch.Tensor,
        faces_path: str,
        faces_url: str,
        visible_cameras: int,
        inward_distance: float,
        outward_distance: float,
        occlusion_eps: float,
        device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``num_points`` body-surface samples into every view.

    Vertices come from ``processed_data/smpl`` and are already posed. They are
    moved into the COLMAP world before sampling, so the cameras can be used as
    stored. The hand is removed with :func:`body_face_mask` on the query frame.
    Occluded samples are marked invalid. Returns pixels ``(V, D, N, 2)`` and a
    bool mask ``(V, D, N)``.
    """

    faces = load_smpl_faces(faces_path, faces_url)
    smpl_dir = os.path.join(sequence, "processed_data", "smpl")
    posed = []
    for frame_index in frame_indices:
        path = os.path.join(smpl_dir, f"{frame_index + 1:05d}.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is missing; surface samples need the posed SMPL meshes")
        item = np.load(path, allow_pickle=True).item()
        if not isinstance(item, dict) or not item:
            raise ValueError(f"{path} does not store a person-to-mesh dict")
        posed.append(item)
    names = sorted(posed[0])
    if any(sorted(item) != names for item in posed):
        raise ValueError("SMPL people changed within the selected frames")
    vertices = torch.tensor(np.stack([
        [item[name]["vertices"] for name in names] for item in posed
    ]), dtype=torch.float32)
    joints = torch.tensor(np.stack([
        posed[init_dataset_index][name]["joints"] for name in names
    ]), dtype=torch.float32)
    if vertices.shape[2:] != (SMPL_VERTEX_COUNT, 3):
        raise ValueError(f"SMPL vertices have shape {tuple(vertices.shape[2:])}, expected ({SMPL_VERTEX_COUNT}, 3)")
    if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
        raise ValueError(f"SMPL joints have shape {tuple(joints.shape)}, expected (people, 45, 3)")
    mesh_device = torch.device("cpu" if device is None else device)
    posed_query = vertices[init_dataset_index]
    face_mask = torch.cat([
        body_face_mask(posed_query[person], joints[person], faces) for person in range(len(names))
    ]).to(mesh_device)
    vertices = (vertices @ transform[:3, :3].T + transform[:3, 3]).to(mesh_device)
    verts = list(vertices.unbind(1))
    mesh_faces = [faces.to(mesh_device)] * len(names)
    warp_device = str(mesh_device)
    eps = float(occlusion_eps * torch.linalg.svdvals(transform[:3, :3]).mean())
    with torch.no_grad():
        binding = sample_near_surface(
            [person[init_dataset_index] for person in verts],
            mesh_faces,
            frames[init_dataset_index],
            num_points,
            visible_cameras,
            inward_distance,
            outward_distance,
            eps,
            warp_device,
            face_mask,
        )
        query_frames = torch.full(
            (binding.xyz.shape[0],), init_dataset_index, dtype=torch.long, device=binding.xyz.device,
        )
        query = Query.from_projection(binding.xyz, frames, query_frames)
        worlds, valid, world_verts, scene_faces, _owner = track_bind_points(verts, mesh_faces, frames, binding)
        points, visibility = project_visibility(
            query, frames, worlds, world_verts, scene_faces, valid, eps, warp_device, binding.face,
        )
    return points, visibility > 0
