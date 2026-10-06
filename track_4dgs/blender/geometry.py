"""World rays and rigid posing in a Blender scene.

Cameras look along +Z, image +Y points downward, and ``R`` / ``T`` use the blend world.
Pixel rays pass through the pixel center ``uv + 0.5``.
"""

from collections.abc import Sequence

import torch
from gaussian_splatting import Camera


def camera_ray(uv: torch.Tensor, camera: Camera) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast one pixel into a unit world ray.

    ``uv`` is ``(2,)`` pixel coordinates in ``(x, y)`` order.
    ``camera.R`` is world-to-camera, so its transpose maps the camera-space direction into the blend world.
    Returns the camera center and a unit direction, each ``(3,)``.
    """
    direction = torch.stack((
        (uv[0] + 0.5 - camera.K[0, 2]) / camera.K[0, 0],
        (uv[1] + 0.5 - camera.K[1, 2]) / camera.K[1, 1],
        uv.new_tensor(1.0),
    ))
    direction = camera.R.T @ direction
    return camera.camera_center, direction / direction.norm()


def triangulate_rays(
    origins: Sequence[torch.Tensor],
    directions: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Intersect world rays by least squares.

    ``origins`` and ``directions`` are equal-length sequences of ``(3,)`` world vectors.
    Directions are unit length.
    Each ray contributes the projection onto the plane perpendicular to its direction.
    Returns the ``(3,)`` point that minimizes squared distance to every ray.
    """
    eye = torch.eye(3, dtype=origins[0].dtype)
    system = torch.zeros((3, 3), dtype=origins[0].dtype)
    target = torch.zeros(3, dtype=origins[0].dtype)
    for origin, direction in zip(origins, directions):
        onto_plane = eye - direction[:, None] * direction[None, :]
        system = system + onto_plane
        target = target + onto_plane @ origin
    return torch.linalg.solve(system, target)


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
