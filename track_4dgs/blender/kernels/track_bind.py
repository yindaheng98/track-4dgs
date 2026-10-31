"""Replay a sampled surface binding on evaluated Blender meshes."""

from collections.abc import Sequence

import torch
from gaussian_splatting.dataset import CameraDataset

from .track import pack_object_faces, scene_vertices
from .sample import SurfaceBinding


def replay_binding(
        world_verts: torch.Tensor,
        scene_faces: torch.Tensor,
        binding: SurfaceBinding) -> torch.Tensor:
    """Move a surface binding onto every posed frame.

    ``world_verts`` is ``(F, P, 3)`` and ``scene_faces`` is ``(T, 3)`` into those vertices.
    Each point keeps ``binding.face``, ``binding.bary``, and ``binding.offset``.
    The face normal is rebuilt from that frame's triangle, so the offset stays along the outward normal.
    Returns ``(F, N, 3)`` world positions.
    The sampled frame reproduces ``binding.xyz``.
    """

    corners = world_verts[:, scene_faces[binding.face.long()].long()]
    edges = torch.cross(corners[:, :, 1] - corners[:, :, 0], corners[:, :, 2] - corners[:, :, 0], dim=-1)
    length = edges.norm(dim=-1).clamp_min(1e-12)
    normal = edges / length.unsqueeze(-1)
    point = (corners * binding.bary.view(1, -1, 3, 1)).sum(dim=2)
    return point + normal * binding.offset.view(1, -1, 1)


def track_bind_points(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor],
        frames: Sequence[CameraDataset],
        binding: SurfaceBinding) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay a surface sample on the evaluated scene.

    ``verts`` lists per-object ``(F, V_i, 3)`` world positions in export order.
    ``faces`` lists per-object ``(T_i, 3)`` triangles into those vertices.
    Vertex count and triangle indices stay fixed across ``F``.
    ``binding`` is a :class:`~track_4dgs.blender.kernels.sample.SurfaceBinding` from the sampled frame.
    Each point keeps that packed face, barycentric weights, and normal offset on every frame.
    Every point is valid.
    Returns ``worlds`` ``(F, N, 3)``, ``valid`` ``(N,)`` int, ``world_verts`` ``(F, P, 3)``,
    ``scene_faces`` ``(T, 3)``, and ``face_owner`` ``(T,)`` long.
    ``worlds`` is the bound point in blend world coordinates.
    ``world_verts`` and ``scene_faces`` are the posed scene those points were bound to.
    ``face_owner`` is the object index of each triangle in ``scene_faces``.
    """
    n_frames = len(frames)
    scene_faces, face_owner = pack_object_faces([obj_verts[0] for obj_verts in verts], faces)
    world_verts = torch.stack([
        scene_vertices([obj_verts[frame_index] for obj_verts in verts])
        for frame_index in range(n_frames)
    ])
    worlds = replay_binding(world_verts, scene_faces, binding)
    valid = torch.ones((worlds.shape[1],), dtype=torch.int32, device=worlds.device)
    return worlds, valid, world_verts, scene_faces, face_owner
