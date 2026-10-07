"""Any-hit visibility from cameras to world points."""

from collections.abc import Sequence

import torch
import warp as wp
from gaussian_splatting.dataset import CameraDataset

from ...tracker import Query
from ...utils import project_points


@wp.kernel
def ray_visibility_kernel(
        mesh: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        points: wp.array(dtype=wp.vec3),
        valid: wp.array(dtype=wp.int32),
        eps: float,
        n_points: int,
        out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    point = tid % n_points
    view = tid // n_points
    if valid[point] == 0:
        out[tid] = 0.0
        return
    delta = points[point] - origins[view]
    dist = wp.length(delta)
    if dist <= eps:
        out[tid] = 1.0
        return
    hit = wp.mesh_query_ray_anyhit(mesh, origins[view], delta / dist, dist - eps)
    out[tid] = 0.0 if hit else 1.0


def ray_visibility(
        camera_centers: torch.Tensor,
        points: torch.Tensor,
        world_verts: torch.Tensor,
        faces: torch.Tensor,
        device: str,
        eps: float,
        valid: torch.Tensor) -> torch.Tensor:
    """Return ``(V, N)`` visibility in ``{0, 1}``.

    ``camera_centers`` is ``(V, 3)`` camera centers.
    ``points`` is ``(N, 3)`` world points.
    ``world_verts`` is ``(P, 3)`` posed vertices, and ``faces`` is ``(T, 3)`` int32 triangles into those vertices.
    ``device`` is a Warp device name such as ``cpu`` or ``cuda``, and Warp is already initialized.
    ``eps`` is the depth tolerance in world units.
    ``valid`` is ``(N,)`` int, and nonzero entries are the points to test.
    A point is visible when the camera-to-point segment does not hit the mesh before ``distance - eps``.
    ``valid == 0`` points are invisible.
    No faces, or a frame with no posed vertices, reports every nonzero ``valid`` point as visible.
    The result lives on ``device``.
    """
    n_views, n_points = camera_centers.shape[0], points.shape[0]
    if faces.numel() == 0 or world_verts.shape[0] == 0 or n_points == 0:
        visible = valid.to(dtype=torch.float32).reshape(1, -1).expand(n_views, n_points).clone()
        return visible.to(device=device)
    world_verts = world_verts.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.reshape(-1).to(device=device, dtype=torch.int32).contiguous()
    camera_centers = camera_centers.to(device=device, dtype=torch.float32).contiguous()
    points = points.to(device=device, dtype=torch.float32).contiguous()
    valid = valid.to(device=device, dtype=torch.int32).contiguous()
    mesh = wp.Mesh(
        points=wp.from_torch(world_verts, dtype=wp.vec3),
        indices=wp.from_torch(faces, dtype=wp.int32),
    )
    visible = wp.zeros(n_views * n_points, dtype=wp.float32, device=device)
    wp.launch(
        ray_visibility_kernel,
        dim=n_views * n_points,
        inputs=[
            mesh.id,
            wp.from_torch(camera_centers, dtype=wp.vec3),
            wp.from_torch(points, dtype=wp.vec3),
            wp.from_torch(valid, dtype=wp.int32),
            float(eps),
            n_points,
        ],
        outputs=[visible],
        device=device,
    )
    return wp.to_torch(visible).reshape(n_views, n_points)


def project_visibility(
        query: Query,
        frames: Sequence[CameraDataset],
        worlds: torch.Tensor,
        world_verts: torch.Tensor,
        scene_faces: torch.Tensor,
        valid: torch.Tensor,
        eps: float,
        device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Project posed points and combine occlusion with the camera frustum.

    ``worlds`` is ``(F, N, 3)`` in blend world coordinates.
    ``world_verts`` is ``(F, P, 3)`` and ``scene_faces`` is ``(T, 3)`` into those vertices.
    ``valid`` is ``(N,)`` and is nonzero for points that were sent to binding.
    ``device`` is a Warp device name, and Warp is already initialized.
    ``eps`` is the depth tolerance in world units.
    Returns ``points`` ``(V, F, N, 2)`` pixels and ``visibility`` ``(V, F, N)``.
    A point is visible when the camera-to-point segment is unoccluded and :func:`project_points` places it in the image.
    """
    n_views = query.points.shape[0]
    n_frames = len(frames)
    n_points = worlds.shape[1]
    points = query.points.new_zeros((n_views, n_frames, n_points, 2))
    visibility = query.points.new_zeros((n_views, n_frames, n_points))
    for frame_index, dataset in enumerate(frames):
        centers = torch.stack([
            dataset[view_index].camera_center.detach().cpu()
            for view_index in range(n_views)
        ])
        ray_visible = ray_visibility(
            centers, worlds[frame_index], world_verts[frame_index], scene_faces, device, eps, valid,
        )
        for view_index in range(n_views):
            camera = dataset[view_index]
            world_xyz = worlds[frame_index].to(device=camera.R.device, dtype=camera.R.dtype)
            uv, in_frustum = project_points(world_xyz, camera)
            hit = ray_visible[view_index].to(device=in_frustum.device)
            points[view_index, frame_index] = uv.to(points.device)
            visibility[view_index, frame_index] = (hit * in_frustum.to(dtype=hit.dtype)).to(visibility.device)
    return points, visibility
