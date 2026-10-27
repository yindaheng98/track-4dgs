"""Sample points in a band around a posed Blender mesh."""

from collections.abc import Sequence

import torch

from ...utils import project_points
from .track import pack_object_faces, scene_vertices


def sample_near_surface(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor],
        cameras: Sequence,
        num_points: int,
        visible_cameras: int,
        distance: float) -> torch.Tensor:
    """Sample ``num_points`` in a band around one frame of the mesh.

    ``verts`` lists world-space ``(V_i, 3)`` vertices in export order.
    Triangles are drawn with probability proportional to area, then each point is moved along the face normal by a uniform offset in ``[-distance, distance]``.
    A point is kept only when it projects into more cameras than ``visible_cameras``.
    The draw is fixed by seed 42.
    """
    world = scene_vertices(verts)
    scene_faces, _ = pack_object_faces(verts, faces)
    triangles = world[scene_faces.long()]
    edges = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    length = edges.norm(dim=-1).clamp_min(1e-12)
    normal = edges / length.unsqueeze(-1)
    area = length * 0.5
    probabilities = area / area.sum()
    n_views = len(cameras)
    if visible_cameras >= n_views:
        raise ValueError("--visible-cameras must be smaller than the number of cameras")
    picked = torch.empty((0, 3), dtype=world.dtype, device=world.device)
    generator = torch.Generator(device=world.device).manual_seed(42)
    while picked.shape[0] < num_points:
        choice = torch.multinomial(probabilities, num_points, replacement=True, generator=generator)
        r1 = torch.rand(num_points, device=world.device, generator=generator).sqrt()
        r2 = torch.rand(num_points, device=world.device, generator=generator)
        barycentric = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], dim=-1)
        point = (barycentric[:, :, None] * triangles[choice]).sum(dim=1)
        offset = (torch.rand(num_points, 1, device=world.device, generator=generator) * 2 - 1) * distance
        xyz = point + normal[choice] * offset
        count = torch.zeros(num_points, dtype=torch.long, device=world.device)
        for view in range(n_views):
            _, in_frustum = project_points(xyz, cameras[view])
            count += in_frustum.to(dtype=torch.long)
        picked = torch.cat([picked, xyz[count > visible_cameras]], dim=0)
    return picked[:num_points]
