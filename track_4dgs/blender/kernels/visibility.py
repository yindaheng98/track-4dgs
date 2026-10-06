"""Any-hit visibility from cameras to world points."""

import torch
import warp as wp


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
        return valid.to(dtype=torch.float32).reshape(1, -1).expand(n_views, n_points).clone()
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
