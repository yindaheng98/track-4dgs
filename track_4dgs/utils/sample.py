"""Point and camera sampling utilities."""

import numpy as np
import torch
from gaussian_splatting.dataset import CameraDataset
from tqdm import tqdm

from .project import project_points


def sample_uniform_xyz(dataset: CameraDataset, num_points: int, visible_cameras: int, device: str) -> torch.Tensor:
    """Sample ``num_points`` uniformly in the cube of radius :meth:`CameraDataset.scene_extent`.

    The cube is centered on the mean camera center. A point is kept only when
    it projects into more cameras than ``visible_cameras``.
    """
    radius = float(dataset.scene_extent())
    n_views = len(dataset)
    center = torch.stack([
        dataset[view].camera_center.detach().to(device=device, dtype=torch.float32)
        for view in range(n_views)
    ]).mean(dim=0)
    lo = center - radius
    hi = center + radius
    if visible_cameras >= n_views:
        raise ValueError("--visible-cameras must be smaller than the number of cameras")
    picked = torch.empty((0, 3), device=device)
    with tqdm(total=num_points, desc="Points", position=1, leave=False) as points_bar:
        while picked.shape[0] < num_points:
            xyz = lo + torch.rand(num_points, 3, device=device) * (hi - lo)
            count = torch.zeros(num_points, dtype=torch.long, device=device)
            for view in range(n_views):
                _, in_frustum = project_points(xyz, dataset[view])
                count += in_frustum.to(dtype=torch.long)
            picked = torch.cat([picked, xyz[count > visible_cameras]], dim=0)
            points_bar.n = min(int(picked.shape[0]), num_points)
            points_bar.refresh()
    return picked[:num_points]


def cap_direction(axis, cap_angle_min, cap_angle_max, rng):
    z = rng.uniform(np.cos(cap_angle_max), np.cos(cap_angle_min))
    phi = rng.uniform(0, 2 * np.pi)
    xy = np.sqrt(1 - z * z)
    local = np.array([xy * np.cos(phi), xy * np.sin(phi), z])
    helper = np.array([0.0, 1.0, 0.0]) if abs(axis[2]) > 0.999 else np.array([0.0, 0.0, 1.0])
    tangent = np.cross(helper, axis)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    return tangent * local[0] + bitangent * local[1] + axis * local[2]


def sample_c2ws(
        n_cameras: int,
        center: tuple[float, float, float],
        radius: tuple[float, float],
        axis: tuple[float, float],
        cap_angle: tuple[float, float],
        spacing: tuple[float, float],
        device: str) -> torch.Tensor:
    """Sample OpenCV camera-to-world poses on a spherical cap.

    Returns ``[V, 4, 4]``. Stops after 100 proposals, so ``V`` may be smaller than ``n_cameras``.
    The draw is fixed by seed 0.
    """

    azimuth, elevation = np.deg2rad(axis)
    axis = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    cap_angle_min, cap_angle_max = np.deg2rad(cap_angle)
    min_angle, max_angle = np.deg2rad(spacing)
    radius_min, radius_max = radius
    center = np.array(center)
    rng = np.random.default_rng(42)

    directions = []
    radii = []
    tries = 0
    while len(directions) < n_cameras:
        tries += 1
        if tries > 100:
            break
        direction = cap_direction(axis, cap_angle_min, cap_angle_max, rng)
        if directions:
            nearest = np.arccos(np.clip(np.stack(directions) @ direction, -1.0, 1.0)).min()
            if nearest < min_angle or nearest > max_angle:
                continue
        directions.append(direction)
        radii.append(rng.uniform(radius_min, radius_max))

    c2ws = []
    for direction, radius in zip(directions, radii):
        position = center + direction * radius
        forward = center - position
        forward /= np.linalg.norm(forward)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up)
        if np.dot(right, right) == 0:
            up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = down
        c2w[:3, 2] = forward
        c2w[:3, 3] = position
        c2ws.append(c2w)
    return torch.tensor(np.stack(c2ws), dtype=torch.float32, device=device)
