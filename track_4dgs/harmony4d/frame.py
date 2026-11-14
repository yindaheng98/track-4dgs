"""The Harmony4D views at one timestep.

Each frame is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
``len`` counts cameras. An image is read when that camera is indexed.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.dataset import LazyCameraDataset
from gaussian_splatting.utils import focal2fov


@dataclass(frozen=True)
class HarmonyCameraSpec:
    """One Harmony4D camera at one timestep. ``load`` reads the image."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: torch.Tensor
    translation: torch.Tensor
    image_path: str
    camera_id: int
    grid: torch.Tensor
    pose_frame: int

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        """Read the image and remap it through :attr:`grid`.

        ``build_camera`` scales the file to ``height`` and ``width`` when they differ.
        """

        camera = build_camera(
            image_width=self.width,
            image_height=self.height,
            FoVx=focal2fov(self.fx, self.width),
            FoVy=focal2fov(self.fy, self.height),
            R=self.rotation,
            T=self.translation,
            cx=self.cx,
            cy=self.cy,
            image_path=self.image_path,
            device=device,
            custom_data={
                "name": self.name,
                "camera_id": self.camera_id,
                "pose_frame": self.pose_frame,
            },
        )
        image = camera.ground_truth_image
        image = F.grid_sample(
            image.unsqueeze(0),
            self.grid.to(device=image.device, dtype=image.dtype).unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        return camera._replace(ground_truth_image=image)


class Harmony4DFrame(LazyCameraDataset):
    """The Harmony4D cameras at one timestep.

    ``camera_specs`` is one :class:`HarmonyCameraSpec` per RGB camera.
    """

    def __init__(self, camera_specs: list[HarmonyCameraSpec], device: str = "cpu"):
        super().__init__(camera_specs, load_mask=False, load_depth=False, device=device)
