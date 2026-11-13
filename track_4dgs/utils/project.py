"""3D-to-2D projection utilities."""

from typing import Tuple

import torch
from gaussian_splatting import Camera


def project(
    xyz: torch.Tensor,
    full_proj_transform: torch.Tensor,
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project ``(K, 3)`` world points with a world-to-clip matrix.

    ``full_proj_transform`` is the 4x4 matrix stored on a camera. The return is
    ``(uv, in_frustum)``: pixel coordinates and whether each point is in front
    of the camera and inside the image.
    """

    p_hom = torch.cat([xyz, xyz.new_ones((xyz.shape[0], 1))], dim=1)
    p_hom = p_hom @ full_proj_transform
    p_w = 1.0 / (p_hom[:, -1:] + 1e-7)
    p_proj = p_hom[:, :-1] * p_w
    W, H = image_width, image_height
    uv = (p_proj[:, :2] + 1.0) * xyz.new_tensor([[W, H]]) * 0.5 - 0.5
    in_frustum = (
        (p_hom[:, -1] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] <= W - 1)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] <= H - 1)
    )
    return uv, in_frustum


def project_points(
    xyz: torch.Tensor,
    camera: Camera,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points into one camera.

    ``xyz`` is ``(K, 3)`` in the camera's world. The camera supplies
    ``full_proj_transform``, ``image_width``, and ``image_height``.
    """

    return project(xyz, camera.full_proj_transform, camera.image_width, camera.image_height)
