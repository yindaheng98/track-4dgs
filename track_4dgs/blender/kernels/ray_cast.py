"""First-hit ray casts against a posed triangle mesh."""

import torch
import warp as wp


MAX_DISTANCE = 1.0e8


@wp.kernel
def ray_cast_kernel(
        mesh: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        directions: wp.array(dtype=wp.vec3),
        max_t: float,
        hit_face: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    query = wp.mesh_query_ray(mesh, origins[tid], directions[tid], max_t)
    if query.result:
        hit_face[tid] = query.face
    else:
        hit_face[tid] = -1


def ray_cast(
        origins: torch.Tensor,
        directions: torch.Tensor,
        world_verts: torch.Tensor,
        faces: torch.Tensor,
        device: str) -> torch.Tensor:
    """Return the first triangle hit by each world ray.

    ``origins`` and ``directions`` are ``(R, 3)``, and directions are unit length.
    ``world_verts`` is ``(P, 3)`` posed vertices, and ``faces`` is ``(T, 3)`` int32 triangles into those vertices.
    ``device`` is a Warp device name such as ``cpu`` or ``cuda``, and Warp is already initialized.
    Returns ``(R,)`` int32 triangle indices on CPU, with ``-1`` for a miss.
    A ray farther than ``MAX_DISTANCE`` misses.
    An empty mesh returns a miss for every ray.
    """
    if faces.numel() == 0 or world_verts.shape[0] == 0 or origins.shape[0] == 0:
        return torch.full((origins.shape[0],), -1, dtype=torch.int32)
    world_verts = world_verts.to(device=device, dtype=torch.float32).contiguous()
    faces = faces.reshape(-1).to(device=device, dtype=torch.int32).contiguous()
    origins = origins.to(device=device, dtype=torch.float32).contiguous()
    directions = directions.to(device=device, dtype=torch.float32).contiguous()
    hit_face = torch.full((origins.shape[0],), -1, dtype=torch.int32, device=device)
    mesh = wp.Mesh(
        points=wp.from_torch(world_verts, dtype=wp.vec3),
        indices=wp.from_torch(faces, dtype=wp.int32),
    )
    wp.launch(
        ray_cast_kernel,
        dim=origins.shape[0],
        inputs=[
            mesh.id,
            wp.from_torch(origins, dtype=wp.vec3),
            wp.from_torch(directions, dtype=wp.vec3),
            MAX_DISTANCE,
        ],
        outputs=[wp.from_torch(hit_face, dtype=wp.int32)],
        device=device,
    )
    return hit_face.cpu()
