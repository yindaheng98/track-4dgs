"""Track query pixels across evaluated Blender meshes."""

from collections.abc import Sequence

import torch
from gaussian_splatting.dataset import CameraDataset

from ...tracker import Query
from .bind import bind_points
from .visibility import project_visibility


def scene_vertices(verts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate one frame of world-space mesh vertices.

    ``verts`` lists ``(V_i, 3)`` vertices in export order.
    Objects with no vertices are skipped.
    Returns ``(sum V_i, 3)`` world vertices.
    A scene with no objects, or only empty objects, returns ``(0, 3)``.
    """
    parts = [obj_verts for obj_verts in verts if obj_verts.shape[0] > 0]
    if not parts:
        reference = verts[0] if verts else None
        if reference is None:
            return torch.zeros((0, 3), dtype=torch.float32)
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
            owners.append(obj_faces.new_full((obj_faces.shape[0],), obj_index, dtype=torch.long))
        offset += obj_verts.shape[0]
    if not parts:
        reference = faces[0] if faces else (verts[0] if verts else None)
        device = None if reference is None else reference.device
        return (
            torch.zeros((0, 3), dtype=torch.int32, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
        )
    return torch.cat(parts), torch.cat(owners)


def track_points(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor],
        query: Query,
        frames: Sequence[CameraDataset],
        eps: float,
        device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bind query pixels to the evaluated scene and project them into every camera.

    ``verts`` lists per-object ``(F, V_i, 3)`` world positions in export order.
    ``faces`` lists per-object ``(T_i, 3)`` triangles into those vertices.
    Vertex count and triangle indices stay fixed across ``F``.
    ``device`` is a Warp device name.
    ``eps`` is the visibility depth tolerance in world units.
    Returns ``points`` ``(V, F, N, 2)``, ``visibility`` ``(V, F, N)``, and ``valid`` ``(N,)`` int.
    """
    n_frames = len(frames)
    scene_faces, face_owner = pack_object_faces([obj_verts[0] for obj_verts in verts], faces)
    if verts:
        world_verts = torch.stack([
            scene_vertices([obj_verts[frame_index] for obj_verts in verts])
            for frame_index in range(n_frames)
        ])
    else:
        world_verts = query.points.new_zeros((n_frames, 0, 3))
    worlds, valid = bind_points(
        world_verts, scene_faces, face_owner, query, frames, device,
    )
    points, visibility = project_visibility(
        query, frames, worlds, world_verts, scene_faces, valid, eps, device,
    )
    return points, visibility, valid
