"""Any-hit visibility of posed Blender meshes."""

import torch
import warp as wp


@wp.kernel
def ray_visibility(
        mesh: wp.uint64,
        origins: wp.array(dtype=wp.vec3),
        targets: wp.array(dtype=wp.vec3),
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
    delta = targets[point] - origins[view]
    dist = wp.length(delta)
    if dist <= eps:
        out[tid] = 1.0
        return
    hit = wp.mesh_query_ray_anyhit(mesh, origins[view], delta / dist, dist - eps)
    out[tid] = 0.0 if hit else 1.0


class MeshOccluder:
    """Any-hit occlusion against one rigid triangle mesh.

    Faces stay fixed for the lifetime of the occluder.
    Each call replaces vertex positions and refits the mesh.
    ``eps`` is subtracted from the segment length so a hit on the target surface itself is not occlusion.
    Warp borrows the vertex and index buffers, so those tensors stay referenced here.
    """

    def __init__(self, faces: torch.Tensor, eps: float):
        self.faces = faces
        self.eps = eps
        self.mesh = None
        self.points = None
        self.indices = None

    def visible(
            self,
            centers: torch.Tensor,
            targets: torch.Tensor,
            world_verts: torch.Tensor,
            valid: torch.Tensor) -> torch.Tensor:
        """Return ``(V, N)`` visibility in ``{0, 1}``.

        ``centers`` is ``(V, 3)`` camera centers.
        ``targets`` is ``(N, 3)`` world points.
        ``world_verts`` is ``(P, 3)`` posed vertices matching ``faces``.
        ``valid`` is ``(N,)`` int, and nonzero entries are the points to test.
        A point is visible when the camera-to-point segment does not hit the mesh before ``distance - eps``.
        ``valid == 0`` points are invisible.
        No faces, or a frame with no posed vertices, reports every nonzero ``valid`` point as visible.
        """
        n_views, n_points = centers.shape[0], targets.shape[0]
        if self.faces.numel() == 0 or world_verts.shape[0] == 0 or n_points == 0:
            return valid.to(dtype=torch.float32).reshape(1, -1).expand(n_views, n_points).clone()
        wp.init()
        device = "cuda" if wp.is_cuda_available() else "cpu"
        torch_device = "cuda" if device == "cuda" else "cpu"
        world_verts = world_verts.to(device=torch_device, dtype=torch.float32).contiguous()
        centers = centers.to(device=torch_device, dtype=torch.float32).contiguous()
        targets = targets.to(device=torch_device, dtype=torch.float32).contiguous()
        valid = valid.to(device=torch_device, dtype=torch.int32).contiguous()
        self.points = world_verts
        posed = wp.from_torch(self.points, dtype=wp.vec3)
        if self.mesh is None:
            self.indices = self.faces.reshape(-1).to(device=torch_device, dtype=torch.int32).contiguous()
            self.mesh = wp.Mesh(
                points=posed,
                indices=wp.from_torch(self.indices, dtype=wp.int32),
            )
        else:
            self.mesh.points = posed
            self.mesh.refit()
        visible = wp.zeros(n_views * n_points, dtype=wp.float32, device=device)
        wp.launch(
            ray_visibility,
            dim=n_views * n_points,
            inputs=[
                self.mesh.id,
                wp.from_torch(centers, dtype=wp.vec3),
                wp.from_torch(targets, dtype=wp.vec3),
                wp.from_torch(valid, dtype=wp.int32),
                float(self.eps),
                n_points,
            ],
            outputs=[visible],
            device=device,
        )
        return wp.to_torch(visible).reshape(n_views, n_points)
