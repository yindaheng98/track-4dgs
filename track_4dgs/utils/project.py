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


def full_proj_transform(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    R: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """World-to-clip matrix for a pinhole camera, in the same layout as a camera's.

    ``R`` and ``T`` are world-to-camera. The near and far planes are ``0.01`` and
    ``100``. Points are row vectors, so the returned matrix is applied on the right.
    """

    R = R.to(dtype=torch.float32)
    T = T.to(device=R.device, dtype=torch.float32)
    znear = 0.01
    zfar = 100.0
    world_view = torch.eye(4, device=R.device, dtype=torch.float32)
    world_view[:3, :3] = R
    world_view[:3, 3] = T
    projection = torch.zeros((4, 4), device=R.device, dtype=torch.float32)
    projection[0, 0] = 2.0 * fx / image_width
    projection[1, 1] = 2.0 * fy / image_height
    projection[0, 2] = 2.0 * cx / image_width - 1.0
    projection[1, 2] = 2.0 * cy / image_height - 1.0
    projection[2, 2] = zfar / (zfar - znear)
    projection[2, 3] = -(zfar * znear) / (zfar - znear)
    projection[3, 2] = 1.0
    return world_view.T @ projection.T


def project_points(
    xyz: torch.Tensor,
    camera: Camera,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points into one camera.

    ``xyz`` is ``(K, 3)`` in the camera's world. The camera supplies
    ``full_proj_transform``, ``image_width``, and ``image_height``.
    """

    return project(xyz, camera.full_proj_transform, camera.image_width, camera.image_height)
