"""3D-to-2D projection utilities using 3DGS full_proj_transform."""

from typing import Tuple

import torch
from gaussian_splatting import Camera


def project_points(
    xyz: torch.Tensor,
    camera: Camera,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points into a single camera's pixel coordinates.

    Uses ``camera.full_proj_transform`` (the 3DGS OpenGL-style
    world-to-clip matrix) for projection.

    Args:
        xyz: ``(K, 3)`` — 3D points in world frame.
        camera: target camera.

    Returns:
        uv: ``(K, 2)`` — ``(x, y)`` pixel coordinates.
        in_frustum: ``(K,)`` bool — ``True`` for points that land
            inside the image and are in front of the camera.
    """
    p_hom = torch.cat([xyz, xyz.new_ones((xyz.shape[0], 1))], dim=1)
    p_hom = p_hom @ camera.full_proj_transform
    p_w = 1.0 / (p_hom[:, -1:] + 1e-7)
    p_proj = p_hom[:, :-1] * p_w
    W, H = camera.image_width, camera.image_height
    uv = (p_proj[:, :2] + 1.0) * xyz.new_tensor([[W, H]]) * 0.5 - 0.5
    in_frustum = (
        (p_hom[:, -1] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= W - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= H - 1)
    )
    return uv, in_frustum
