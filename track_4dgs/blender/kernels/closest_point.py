"""Closest-point snaps onto one posed triangle mesh."""

import torch
import warp as wp


MAX_DISTANCE = 1.0e8


@wp.kernel
def closest_point_kernel(
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: float,
        out: wp.array(dtype=wp.vec3),
        ok: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    query = wp.mesh_query_point(mesh, points[tid], max_dist)
    if query.result:
        out[tid] = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        ok[tid] = 1
    else:
        ok[tid] = 0


def closest_point(
        points: torch.Tensor,
        world_verts: torch.Tensor,
        faces: torch.Tensor,
        device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap world points onto one posed mesh.

    ``points`` is ``(K, 3)`` in world coordinates.
    ``world_verts`` is ``(V, 3)`` posed vertices, and ``faces`` is ``(T, 3)`` int32 triangles into those vertices.
    ``device`` is a Warp device name such as ``cpu`` or ``cuda``.
    Returns ``ok`` ``(K,)`` bool and ``snapped`` ``(K, 3)`` world points, both on the device of ``points``.
    A point farther than ``MAX_DISTANCE`` from the mesh is left unsnapped.
    A mesh with no triangles leaves every point unsnapped.
    """
    if faces.numel() == 0 or world_verts.shape[0] == 0:
        return (
            torch.zeros(points.shape[0], dtype=torch.bool, device=points.device),
            torch.zeros((points.shape[0], 3), dtype=points.dtype, device=points.device),
        )
    world_verts = world_verts.contiguous()
    faces = faces.reshape(-1).contiguous()
    points = points.contiguous()
    snapped = torch.zeros_like(points)
    ok = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
    wp.init()
    mesh = wp.Mesh(
        points=wp.from_torch(world_verts, dtype=wp.vec3),
        indices=wp.from_torch(faces, dtype=wp.int32),
    )
    wp.launch(
        closest_point_kernel,
        dim=points.shape[0],
        inputs=[mesh.id, wp.from_torch(points, dtype=wp.vec3), MAX_DISTANCE],
        outputs=[wp.from_torch(snapped, dtype=wp.vec3), wp.from_torch(ok, dtype=wp.int32)],
        device=device,
    )
    return ok.bool(), snapped
