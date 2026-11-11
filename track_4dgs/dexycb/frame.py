"""The DexYCB views at one timestep.

Each frame is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
``len`` counts cameras. An image is read when that camera is indexed.
"""

from dataclasses import dataclass

import torch
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.dataset import LazyCameraDataset
from gaussian_splatting.utils import focal2fov

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


@dataclass(frozen=True)
class DexYCBCameraSpec:
    """One DexYCB color camera at one timestep. ``load`` reads the image."""

    serial: str
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: torch.Tensor
    translation: torch.Tensor
    image_path: str
    depth_path: str

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        return build_camera(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            FoVx=focal2fov(self.fx, IMAGE_WIDTH),
            FoVy=focal2fov(self.fy, IMAGE_HEIGHT),
            R=self.rotation,
            T=self.translation,
            cx=self.cx,
            cy=self.cy,
            image_path=self.image_path,
            depth_path=self.depth_path if load_depth else None,
            device=device,
            custom_data={"serial": self.serial},
            resize_if_mismatch=None,
        )


class DexYCBFrame(LazyCameraDataset):
    """The DexYCB views at one timestep.

    ``camera_specs`` is one :class:`DexYCBCameraSpec` per color camera.
    """

    def __init__(self, camera_specs: list[DexYCBCameraSpec], load_depth: bool, device: str = "cpu"):
        super().__init__(camera_specs, load_mask=False, load_depth=load_depth, device=device)
