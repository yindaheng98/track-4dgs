"""Point sampling utilities."""

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
