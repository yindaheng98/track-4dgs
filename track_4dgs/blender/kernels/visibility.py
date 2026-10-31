"""Camera-to-point visibility on a posed triangle mesh."""

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
        n_points: int,
        hit_face: wp.array(dtype=wp.int32),
        hit_distance: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    point = tid % n_points
    view = tid // n_points
    delta = points[point] - origins[view]
    dist = wp.length(delta)
    hit_face[tid] = -1
    hit_distance[tid] = dist
    if dist <= 0.0:
        return
    query = wp.mesh_query_ray(mesh, origins[view], delta / dist, dist)
    if query.result:
        hit_face[tid] = query.face
        hit_distance[tid] = query.t


def ray_visibility(
        camera_centers: torch.Tensor,
        points: torch.Tensor,
        world_verts: torch.Tensor,
        faces: torch.Tensor,
        device: str,
        occlusion_eps: float,
        valid: torch.Tensor,
        ignore_face: torch.Tensor | None = None) -> torch.Tensor:
    """Return ``(V, N)`` visibility in ``{0, 1}``.

    ``camera_centers`` is ``(V, 3)`` camera centers.
    ``points`` is ``(N, 3)`` world points.
    ``world_verts`` is ``(P, 3)`` posed vertices, and ``faces`` is ``(T, 3)`` int32 triangles into those vertices.
    ``device`` is a Warp device name such as ``cpu`` or ``cuda``.
    ``occlusion_eps`` is the occlusion depth tolerance in world units.
    ``valid`` is ``(N,)`` int, and nonzero entries are the points to test.
    ``ignore_face`` is ``(N,)`` int. Negative entries exempt nothing. The default exempts nothing.
    The closest hit does not occlude when its face is ``ignore_face`` or it lies within ``occlusion_eps`` of the point.
    Triangles behind that hit are not tested.
    ``valid == 0`` points are invisible.
    No faces, or a frame with no posed vertices, reports every nonzero ``valid`` point as visible.
    The result stays on the device of the inputs.
    """
    n_views, n_points = camera_centers.shape[0], points.shape[0]
    if faces.numel() == 0 or world_verts.shape[0] == 0 or n_points == 0:
        return valid.to(dtype=torch.float32).reshape(1, -1).expand(n_views, n_points).clone()
    world_verts = world_verts.contiguous()
    faces = faces.reshape(-1).contiguous()
    camera_centers = camera_centers.contiguous()
    points = points.contiguous()
    valid = valid.contiguous()
    if ignore_face is None:
        ignore_face = torch.full((n_points,), -1, dtype=torch.int32, device=points.device)
    else:
        ignore_face = ignore_face.to(device=points.device, dtype=torch.int32).reshape(-1).contiguous()
    wp.init()
    mesh = wp.Mesh(
        points=wp.from_torch(world_verts, dtype=wp.vec3),
        indices=wp.from_torch(faces, dtype=wp.int32),
    )
    hit_face = wp.empty(n_views * n_points, dtype=wp.int32, device=device)
    hit_distance = wp.empty(n_views * n_points, dtype=wp.float32, device=device)
    wp.launch(
        ray_visibility_kernel,
        dim=n_views * n_points,
        inputs=[
            mesh.id,
            wp.from_torch(camera_centers, dtype=wp.vec3),
            wp.from_torch(points, dtype=wp.vec3),
            n_points,
        ],
        outputs=[hit_face, hit_distance],
        device=device,
    )
    face = wp.to_torch(hit_face).reshape(n_views, n_points)
    distance = wp.to_torch(hit_distance).reshape(n_views, n_points)
    point_distance = torch.linalg.norm(points - camera_centers[:, None, :], dim=-1)
    near_point = point_distance - distance <= occlusion_eps
    ignored = face == ignore_face.view(1, -1)
    visible = (face < 0) | ignored | near_point
    visible = visible & valid.to(dtype=torch.bool).view(1, -1)
    return visible.to(dtype=points.dtype)


def project_visibility(
        query: Query,
        frames: Sequence[CameraDataset],
        worlds: torch.Tensor,
        world_verts: torch.Tensor,
        scene_faces: torch.Tensor,
        valid: torch.Tensor,
        occlusion_eps: float,
        device: str,
        ignore_face: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Project posed points and combine occlusion with the camera frustum.

    ``worlds`` is ``(F, N, 3)`` in blend world coordinates.
    ``world_verts`` is ``(F, P, 3)`` and ``scene_faces`` is ``(T, 3)`` into those vertices.
    ``valid`` is ``(N,)`` and is nonzero for points that were sent to binding.
    ``device`` is a Warp device name.
    ``occlusion_eps`` is the occlusion depth tolerance in world units.
    ``ignore_face`` is an optional ``(N,)`` int passed to :func:`ray_visibility`.
    The default counts every triangle as an occluder.
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
            dataset[view_index].camera_center.detach()
            for view_index in range(n_views)
        ])
        ray_visible = ray_visibility(
            centers, worlds[frame_index], world_verts[frame_index], scene_faces, device, occlusion_eps, valid, ignore_face,
        )
        for view_index in range(n_views):
            camera = dataset[view_index]
            uv, in_frustum = project_points(worlds[frame_index], camera)
            hit = ray_visible[view_index]
            points[view_index, frame_index] = uv
            visibility[view_index, frame_index] = hit * in_frustum.to(dtype=hit.dtype)
    return points, visibility
