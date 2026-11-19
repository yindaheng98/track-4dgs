"""The Panoptic HD views at one timestep.

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
class PanopticCameraSpec:
    """One Panoptic HD camera at one timestep. ``load`` reads and undistorts the image.

    ``width`` and the intrinsics are the image handed to the camera. They match
    the calibration until ``max_image_size`` scales the long side down.
    ``grid`` undistorts ``ground_truth_image`` after ``build_camera`` has loaded
    and resized it. ``image_path`` is ``hdImgs/<name>/<name>_XXXXXXXX.png``.
    ``frame_index`` is that HD frame.
    """

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: torch.Tensor
    translation: torch.Tensor
    grid: torch.Tensor
    image_path: str
    frame_index: int

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        """Load the photo with ``build_camera``, then remap that image through :attr:`grid`.

        ``build_camera`` scales the file to ``height`` and ``width`` when they differ.
        ``grid`` comes from :func:`~track_4dgs.utils.opencv_undistort_grid` and is
        sampled the same way as a Harmony4D fisheye grid.
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
            custom_data={"name": self.name, "frame_index": self.frame_index},
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


class PanopticFrame(LazyCameraDataset):
    """The Panoptic HD cameras at one timestep.

    ``camera_specs`` is one :class:`PanopticCameraSpec` per HD camera.
    """

    def __init__(self, camera_specs: list[PanopticCameraSpec], device: str = "cpu"):
        super().__init__(camera_specs, load_mask=False, load_depth=False, device=device)
