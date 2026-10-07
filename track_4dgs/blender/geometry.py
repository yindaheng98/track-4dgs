"""Rigid posing of Blender object-space meshes."""

from collections.abc import Sequence

import torch


def object_points_to_world(
    local: torch.Tensor,
    matrices: torch.Tensor,
    obj_index: torch.Tensor,
) -> torch.Tensor:
    """Apply each frame's ``matrix_world`` to object-local points.

    ``local`` is ``(N, 3)``.
    ``matrices`` is ``(F, M, 4, 4)`` in export object order.
    ``obj_index`` is ``(N,)`` long.
    Index ``-1`` keeps the point as a world position already stored in ``local``.
    Returns ``(F, N, 3)`` world points.
    """
    hom = torch.cat([local, torch.ones(local.shape[0], 1)], dim=1)
    worlds = hom[:, :3].unsqueeze(0).expand(matrices.shape[0], -1, -1).clone()
    bound = obj_index >= 0
    if bool(bound.any()):
        posed = torch.einsum("fbij,bj->fbi", matrices[:, obj_index[bound]], hom[bound])
        worlds[:, bound] = posed[..., :3]
    return worlds


def posed_scene_vertices(
    verts: Sequence[torch.Tensor],
    matrix_world: torch.Tensor,
) -> torch.Tensor:
    """Concatenate one frame of rigidly posed mesh vertices.

    ``verts`` lists object-local ``(V_i, 3)`` vertices in export order.
    ``matrix_world`` is ``(M, 4, 4)`` for that frame, in the same order.
    Objects with no vertices are skipped.
    Returns ``(sum V_i, 3)`` world vertices, or ``(0, 3)`` when every object is empty.
    """
    parts = []
    for obj_verts, matrix in zip(verts, matrix_world):
        if obj_verts.shape[0] == 0:
            continue
        hom = torch.cat([obj_verts, torch.ones(obj_verts.shape[0], 1)], dim=1)
        parts.append(torch.einsum("ij,vj->vi", matrix, hom)[:, :3])
    if not parts:
        return verts[0].new_zeros((0, 3))
    return torch.cat(parts)
