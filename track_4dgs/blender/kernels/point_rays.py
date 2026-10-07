"""World rays through Blender cameras.

Cameras look along +Z, image +Y points downward, and ``R`` / ``T`` use the blend world.
Pixel rays pass through the pixel center ``uv + 0.5``.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from gaussian_splatting import Camera


def camera_ray(uv: torch.Tensor, cameras: Sequence[Camera]) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast pixels into unit world rays.

    ``uv`` is ``(V, ..., 2)`` pixel coordinates in ``(x, y)`` order, one leading row per camera.
    ``camera.R`` is world-to-camera, so its transpose maps the camera-space direction into the blend world.
    Returns origins and directions, each ``(V, ..., 3)``, on CPU.
    """
    intrinsics = torch.stack([camera.K.detach().cpu() for camera in cameras])
    rotations = torch.stack([camera.R.detach().cpu() for camera in cameras])
    centers = torch.stack([camera.camera_center.detach().cpu() for camera in cameras])
    uv = uv.detach().cpu().to(dtype=intrinsics.dtype)
    focal = torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=-1)
    principal = torch.stack([intrinsics[:, 0, 2], intrinsics[:, 1, 2]], dim=-1)
    broadcast = (uv.shape[0],) + (1,) * (uv.ndim - 2) + (2,)
    xy = (uv + 0.5 - principal.view(broadcast)) / focal.view(broadcast)
    direction = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    direction = torch.einsum("vij,v...j->v...i", rotations.transpose(-1, -2), direction)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    origins = centers.view(broadcast[:-1] + (3,)).expand_as(direction)
    return origins, direction


def triangulate_rays(
    origins: torch.Tensor,
    directions: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Intersect world rays by least squares.

    ``origins`` and ``directions`` are ``(..., S, 3)``.
    Directions are unit length.
    ``mask`` is ``(..., S)`` bool and selects the rays that enter the solve.
    ``None`` uses every ray.
    Each selected ray contributes the projection onto the plane perpendicular to its direction.
    Returns the ``(..., 3)`` points that minimize squared distance to the selected rays.
    """
    eye = torch.eye(3, dtype=origins.dtype, device=origins.device)
    onto_plane = eye - directions[..., :, None] * directions[..., None, :]
    if mask is not None:
        onto_plane = onto_plane * mask[..., None, None].to(dtype=onto_plane.dtype)
    system = onto_plane.sum(dim=-3)
    target = torch.matmul(onto_plane, origins[..., None]).squeeze(-1).sum(dim=-2)
    return torch.linalg.solve(system, target)


@dataclass(frozen=True)
class PointRays:
    """Triangulated points and the world rays cast from one set of cameras.

    ``xyz`` is ``(N, 3)``.
    ``origins`` and ``directions`` are ``(N, V, 3)`` unit world rays.
    ``ray_mask`` is ``(N, V)`` bool and selects the rays that belong to each point.
    A point with fewer than two selected rays keeps ``xyz`` at zero.
    """

    xyz: torch.Tensor
    origins: torch.Tensor
    directions: torch.Tensor
    ray_mask: torch.Tensor

    @classmethod
    def from_cameras(
            cls,
            uv: torch.Tensor,
            cameras: Sequence[Camera],
            mask: torch.Tensor) -> 'PointRays':
        """Cast ``uv`` through ``cameras`` and triangulate the selected rays.

        ``uv`` is ``(V, N, 2)`` pixel coordinates, one leading row per camera.
        ``mask`` is ``(V, N)`` bool.
        ``True`` selects a view for that point.
        """
        origins, directions = camera_ray(uv, cameras)
        origins = origins.transpose(0, 1).contiguous()
        directions = directions.transpose(0, 1).contiguous()
        ray_mask = mask.detach().cpu().transpose(0, 1).contiguous()
        valid = ray_mask.sum(dim=-1) >= 2
        xyz = torch.zeros(origins.shape[0], 3, dtype=origins.dtype, device=origins.device)
        if bool(valid.any()):
            xyz[valid] = triangulate_rays(origins[valid], directions[valid], ray_mask[valid])
        return cls(xyz=xyz, origins=origins, directions=directions, ray_mask=ray_mask)

    @property
    def valid(self) -> torch.Tensor:
        """Return ``(N,)`` bool, true when a point has at least two rays."""
        return self.ray_mask.sum(dim=-1) >= 2
