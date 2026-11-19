"""The Panoptic HD views at one timestep.

Each frame is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
``len`` counts cameras. An image is read when that camera is indexed.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.dataset import LazyCameraDataset
from gaussian_splatting.utils import focal2fov, read_image


def read_frame(path: str, frame_index: int, from_video: bool) -> torch.Tensor:
    """``(3, H, W)`` RGB float image in ``[0, 1]``.

    An extracted frame is read as an image. A video uses frame ``frame_index``.
    """

    if not from_video:
        return read_image(path)
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise FileNotFoundError(path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise FileNotFoundError(f"{path} has no frame {frame_index}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(dtype=torch.float32) / 255.0


@dataclass(frozen=True)
class PanopticCameraSpec:
    """One Panoptic HD camera at one timestep. ``load`` reads and undistorts the image.

    ``width`` and the intrinsics are the image handed to the camera. They match
    the calibration until ``max_image_size`` scales the long side down.
    ``grid`` undistorts the photo after it is scaled to that size.
    ``image_path`` is an extracted frame or ``hdVideos/hd_PP_NN.mp4``.
    ``frame_index`` selects the video frame and is unused for an extracted image.
    The camera's ``ground_truth_image_path`` stays ``None``: the photo comes from
    :func:`read_frame`, not from ``build_camera``.
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
    from_video: bool

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        """Read one frame, scale it, remap it through :attr:`grid`, and attach it.

        ``build_camera`` is not given a path, so ``ground_truth_image_path`` is
        ``None``. The resize uses ``align_corners=False``, matching ``build_camera``.
        ``grid`` comes from :func:`~track_4dgs.utils.opencv_undistort_grid` and is
        sampled the same way as a Harmony4D fisheye grid.
        """

        image = read_frame(self.image_path, self.frame_index, self.from_video)
        if image.shape[1:] != (self.height, self.width):
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        image = F.grid_sample(
            image.unsqueeze(0),
            self.grid.to(device=image.device, dtype=image.dtype).unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        camera = build_camera(
            image_width=self.width,
            image_height=self.height,
            FoVx=focal2fov(self.fx, self.width),
            FoVy=focal2fov(self.fy, self.height),
            R=self.rotation,
            T=self.translation,
            cx=self.cx,
            cy=self.cy,
            device=device,
            custom_data={"name": self.name, "frame_index": self.frame_index},
        )
        return camera._replace(
            ground_truth_image=image.to(device=camera.R.device, dtype=camera.R.dtype),
        )


class PanopticFrame(LazyCameraDataset):
    """The Panoptic HD cameras at one timestep.

    ``camera_specs`` is one :class:`PanopticCameraSpec` per HD camera.
    """

    def __init__(self, camera_specs: list[PanopticCameraSpec], device: str = "cpu"):
        super().__init__(camera_specs, load_mask=False, load_depth=False, device=device)
