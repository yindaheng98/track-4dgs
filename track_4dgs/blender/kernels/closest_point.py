"""Closest-point snaps onto one posed triangle mesh."""

import torch
import warp as wp


MAX_DISTANCE = 1.0e8


@wp.kernel
def closest_point_kernel(
        mesh: wp.uint64,
        points: wp.array(dtype=wp.vec3),
        max_dist: float,
        face: wp.array(dtype=wp.int32),
        uv: wp.array(dtype=wp.vec2),
        ok: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    query = wp.mesh_query_point(mesh, points[tid], max_dist)
    if query.result:
        face[tid] = query.face
        uv[tid] = wp.vec2(query.u, query.v)
        ok[tid] = 1
    else:
        face[tid] = -1
        uv[tid] = wp.vec2(0.0, 0.0)
        ok[tid] = 0


def closest_point(
        points: torch.Tensor,
        world_verts: torch.Tensor,
        faces: torch.Tensor,
        device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Snap world points onto one posed mesh.

    ``points`` is ``(K, 3)`` in world coordinates.
    ``world_verts`` is ``(V, 3)`` posed vertices, and ``faces`` is ``(T, 3)`` int32 triangles into those vertices.
    ``device`` is a Warp device name such as ``cpu`` or ``cuda``.
    Returns ``ok`` ``(K,)`` bool, ``face`` ``(K,)`` int32, and ``uv`` ``(K, 2)``, all on the device of ``points``.
    ``face`` indexes ``faces``.
    A miss is ``-1``.
    ``uv`` is Warp's barycentric pair for that triangle.
    The snapped point is ``u * v0 + v * v1 + (1 - u - v) * v2``.
    A point farther than ``MAX_DISTANCE`` from the mesh is left unsnapped.
    A mesh with no triangles leaves every point unsnapped.
    """
    misses = (
        torch.zeros(points.shape[0], dtype=torch.bool, device=points.device),
        torch.full((points.shape[0],), -1, dtype=torch.int32, device=points.device),
        torch.zeros((points.shape[0], 2), dtype=points.dtype, device=points.device),
    )
    if faces.numel() == 0 or world_verts.shape[0] == 0 or points.shape[0] == 0:
        return misses
    world_verts = world_verts.contiguous()
    faces = faces.reshape(-1).contiguous()
    points = points.contiguous()
    face = misses[1].clone()
    uv = misses[2].clone()
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
        outputs=[
            wp.from_torch(face, dtype=wp.int32),
            wp.from_torch(uv, dtype=wp.vec2),
            wp.from_torch(ok, dtype=wp.int32),
        ],
        device=device,
    )
    return ok.bool(), face, uv
