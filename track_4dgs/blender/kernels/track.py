"""Pose a rigid Blender scene and track query pixels through it."""

from collections.abc import Sequence

import torch
from gaussian_splatting.dataset import CameraDataset

from ...tracker import Query
from .bind import bind_points
from .visibility import project_visibility


def posed_scene_vertices(
        verts: Sequence[torch.Tensor],
        matrix_world: torch.Tensor) -> torch.Tensor:
    """Concatenate one frame of rigidly posed mesh vertices.

    ``verts`` lists object-local ``(V_i, 3)`` vertices in export order.
    ``matrix_world`` is ``(M, 4, 4)`` for that frame, in the same order.
    Objects with no vertices are skipped.
    Returns ``(sum V_i, 3)`` world vertices.
    A scene with no objects, or only empty objects, returns ``(0, 3)``.
    """
    parts = []
    for obj_verts, matrix in zip(verts, matrix_world):
        if obj_verts.shape[0] == 0:
            continue
        hom = torch.cat([obj_verts, torch.ones(obj_verts.shape[0], 1)], dim=1)
        parts.append(torch.einsum("ij,vj->vi", matrix, hom)[:, :3])
    if not parts:
        reference = verts[0] if verts else matrix_world
        return reference.new_zeros((0, 3))
    return torch.cat(parts)


def pack_object_faces(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate per-object triangles and record which object owns each one.

    A face index is shifted by the vertex count of every preceding object.
    Returns ``faces`` ``(T, 3)`` int32 and ``owner`` ``(T,)`` long.
    """
    parts = []
    owners = []
    offset = 0
    for obj_index, (obj_verts, obj_faces) in enumerate(zip(verts, faces)):
        if obj_faces.numel() > 0:
            parts.append(obj_faces + offset)
            owners.append(torch.full((obj_faces.shape[0],), obj_index, dtype=torch.long))
        offset += obj_verts.shape[0]
    if not parts:
        return torch.zeros((0, 3), dtype=torch.int32), torch.zeros((0,), dtype=torch.long)
    return torch.cat(parts), torch.cat(owners)


def track_points(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor],
        matrices: torch.Tensor,
        query: Query,
        frames: Sequence[CameraDataset],
        eps: float,
        device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind query pixels to the rigid scene and project them into every camera.

    ``verts`` and ``faces`` are per-object tensors in export order.
    ``matrices`` is ``(F, M, 4, 4)`` ``matrix_world``.
    ``device`` is a Warp device name, and Warp is already initialized.
    ``eps`` is the visibility depth tolerance in world units.
    Returns ``points`` ``(V, F, N, 2)``, ``visibility`` ``(V, F, N)``, and ``valid`` ``(N,)`` int.
    """
    scene_faces, face_owner = pack_object_faces(verts, faces)
    world_verts = torch.stack([
        posed_scene_vertices(verts, matrices[frame_index]) for frame_index in range(matrices.shape[0])
    ])
    worlds, valid = bind_points(
        world_verts, scene_faces, face_owner, query, frames, matrices, device,
    )
    points, visibility = project_visibility(
        query, frames, worlds, world_verts, scene_faces, valid, eps, device,
    )
    return points, visibility, valid
