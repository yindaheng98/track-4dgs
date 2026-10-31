"""Sample points in a band around a posed Blender mesh."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from gaussian_splatting.camera import Camera

from ...utils import project_points
from .track import pack_object_faces, scene_vertices
from .visibility import ray_visibility


@dataclass(frozen=True)
class SurfaceBinding:
    """Points tied to packed scene triangles on the frame they were sampled.

    ``xyz`` is ``(N, 3)`` world positions, ``point + normal * offset``.
    ``face`` is ``(N,)`` long indices into the packed scene triangles from :func:`~track_4dgs.blender.kernels.track.pack_object_faces`.
    ``bary`` is ``(N, 3)`` and sums to one. It weights the three triangle vertices in order.
    ``offset`` is ``(N,)`` signed distance along the outward face normal.
    A negative offset lies inside the mesh.
    """

    xyz: torch.Tensor
    face: torch.Tensor
    bary: torch.Tensor
    offset: torch.Tensor


def sample_near_surface(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor],
        cameras: Sequence[Camera],
        num_points: int,
        visible_cameras: int,
        inward_distance: float,
        outward_distance: float,
        occlusion_eps: float,
        device: str,
        face_mask: torch.Tensor | None = None,
        max_reject_draws: int = 16) -> SurfaceBinding:
    """Sample ``num_points`` in a band around one frame of the mesh.

    ``verts`` lists world-space ``(V_i, 3)`` vertices in export order.
    ``verts`` and ``faces`` are the complete occluding scene.
    ``face_mask`` is an optional ``(T,)`` bool, one entry per triangle from :func:`~track_4dgs.blender.kernels.track.pack_object_faces`.
    ``True`` allows that triangle to be sampled. ``None`` samples every packed face.
    A selection with no positive area raises ``ValueError``.
    Triangles are drawn with probability proportional to area, using only the selected faces.
    A point on the chosen triangle uses barycentric weights ``(1 - r1, r1 * (1 - r2), r1 * r2)`` with ``r1 = sqrt(u)``.
    It is then moved along the outward face normal by a uniform offset in ``[-inward_distance, outward_distance]``.
    A negative offset goes inward.
    A point is kept only when more cameras than ``visible_cameras`` see it.
    A camera sees it when the point projects into the image and the closest hit on the camera-to-point segment is not occlusion.
    That hit is not occlusion when it is the bound triangle or lies within ``occlusion_eps`` of the point.
    Triangles behind that hit are not tested.
    Candidates are drawn again until ``num_points`` survive.
    ``max_reject_draws`` draws in a row that keep nothing raise ``ValueError``. It defaults to 16.
    That happens when the faces seen by more than ``visible_cameras`` cameras cannot supply the sample.
    ``occlusion_eps`` is that occlusion depth tolerance in world units, and ``device`` is a Warp device name.
    The returned binding stores the packed complete-scene face, the barycentric weights, and that signed offset.
    Visibility rays test every packed face. The chosen face is ignored as an occluder.
    The draw is fixed by seed 42.
    """
    world = scene_vertices(verts)
    scene_faces, _ = pack_object_faces(verts, faces)
    triangles = world[scene_faces.long()]
    edges = torch.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1)
    raw_length = edges.norm(dim=-1)
    length = raw_length.clamp_min(1e-12)
    normal = edges / length.unsqueeze(-1)
    area = length * 0.5
    if face_mask is None:
        probabilities = area / area.sum()
    else:
        selected = face_mask.to(device=raw_length.device)
        if raw_length[selected].sum() == 0:
            raise ValueError("face_mask selects no triangle with positive area")
        selected_area = area.masked_fill(~selected, 0)
        probabilities = selected_area / selected_area.sum()
    n_views = len(cameras)
    if visible_cameras >= n_views:
        raise ValueError("--visible-cameras must be smaller than the number of cameras")
    if inward_distance < 0 or outward_distance < 0:
        raise ValueError("inward_distance and outward_distance must be non-negative")
    if max_reject_draws < 1:
        raise ValueError("max_reject_draws must be positive")
    picked_xyz = torch.empty((0, 3), dtype=world.dtype, device=world.device)
    picked_face = torch.empty((0,), dtype=torch.long, device=world.device)
    picked_bary = torch.empty((0, 3), dtype=world.dtype, device=world.device)
    picked_offset = torch.empty((0,), dtype=world.dtype, device=world.device)
    centers = torch.stack([
        cameras[view].camera_center.detach().to(device=world.device, dtype=world.dtype)
        for view in range(n_views)
    ])
    valid = torch.ones(num_points, dtype=torch.int32, device=world.device)
    generator = torch.Generator(device=world.device).manual_seed(42)
    draws_without_keep = 0
    while picked_xyz.shape[0] < num_points:
        choice = torch.multinomial(probabilities, num_points, replacement=True, generator=generator)
        r1 = torch.rand(num_points, device=world.device, generator=generator).sqrt()
        r2 = torch.rand(num_points, device=world.device, generator=generator)
        barycentric = torch.stack([1 - r1, r1 * (1 - r2), r1 * r2], dim=-1)
        point = (barycentric[:, :, None] * triangles[choice]).sum(dim=1)
        offset = torch.rand(num_points, device=world.device, generator=generator) * (inward_distance + outward_distance) - inward_distance
        xyz = point + normal[choice] * offset[:, None]
        ray_visible = ray_visibility(centers, xyz, world, scene_faces, device, occlusion_eps, valid, choice)
        count = torch.zeros(num_points, dtype=torch.long, device=world.device)
        for view in range(n_views):
            _, in_frustum = project_points(xyz, cameras[view])
            seen = ray_visible[view] * in_frustum.to(dtype=ray_visible.dtype)
            count += seen.to(dtype=torch.long)
        keep = count > visible_cameras
        if not bool(keep.any()):
            draws_without_keep += 1
            if draws_without_keep >= max_reject_draws:
                raise ValueError(
                    f"kept {picked_xyz.shape[0]} points seen by more than {visible_cameras} cameras "
                    f"after {max_reject_draws} draws kept nothing; {num_points} were requested")
            continue
        draws_without_keep = 0
        picked_xyz = torch.cat([picked_xyz, xyz[keep]], dim=0)
        picked_face = torch.cat([picked_face, choice[keep]], dim=0)
        picked_bary = torch.cat([picked_bary, barycentric[keep]], dim=0)
        picked_offset = torch.cat([picked_offset, offset[keep]], dim=0)
    return SurfaceBinding(
        xyz=picked_xyz[:num_points],
        face=picked_face[:num_points],
        bary=picked_bary[:num_points],
        offset=picked_offset[:num_points],
    )
