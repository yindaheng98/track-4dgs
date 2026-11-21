"""The Panoptic HD views at one timestep.

Each frame is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
``len`` counts cameras. An image is read when that camera is indexed.
"""

from dataclasses import dataclass

import cv2
import torch
import torch.nn.functional as F
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.dataset import LazyCameraDataset
from gaussian_splatting.utils import focal2fov


@dataclass(frozen=True)
class PanopticCameraSpec:
    """One Panoptic HD camera at one timestep.

    ``width`` and the intrinsics are the image handed to the camera. They match
    the calibration until ``max_image_size`` scales the long side down.
    ``grid`` undistorts the decoded frame after it has been scaled to ``height``
    and ``width``. ``video_path`` is that camera's HD video. ``frame_index`` is
    the HD frame inside it, starting at 0.
    :meth:`load_with_image` undistorts a frame already in memory.
    :meth:`load` reads that frame from the video first. Its arguments match
    :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
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
    video_path: str
    frame_index: int

    def load_with_image(self, image: torch.Tensor) -> Camera:
        """Undistort ``image`` and return its camera.

        ``image`` is ``(3, H, W)`` float32 RGB in ``[0, 1]``, already on the
        camera device. A spatial size other than ``height`` by ``width`` is
        resized with ``align_corners=False``, the same mapping
        :func:`~gaussian_splatting.camera.build_camera` uses for a file.
        ``grid`` comes from :func:`~track_4dgs.utils.opencv_undistort_grid`.
        """

        if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.float32:
            raise ValueError(
                f"{self.video_path} frame {self.frame_index} image is {tuple(image.shape)} {image.dtype}, expected (3, H, W) float32"
            )
        camera = build_camera(
            image_width=self.width,
            image_height=self.height,
            FoVx=focal2fov(self.fx, self.width),
            FoVy=focal2fov(self.fy, self.height),
            R=self.rotation,
            T=self.translation,
            cx=self.cx,
            cy=self.cy,
            device=image.device,
            custom_data={"name": self.name, "frame_index": self.frame_index},
        )
        if image.shape[1:] != (self.height, self.width):
            image = F.interpolate(
                image.unsqueeze(0), size=(self.height, self.width), mode="bilinear", align_corners=False,
            ).squeeze(0)
        image = F.grid_sample(
            image.unsqueeze(0),
            self.grid.to(device=image.device, dtype=image.dtype).unsqueeze(0),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)
        return camera._replace(ground_truth_image=image, ground_truth_image_path=self.video_path)

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        """Read this frame from the video and undistort it.

        ``load_mask`` and ``load_depth`` are accepted for
        :class:`~gaussian_splatting.dataset.LazyCameraDataset` and are not used.
        Decoding starts at frame 0 and stops at :attr:`frame_index`. The kept
        frame is RGB float32 in ``[0, 1]`` on ``device``.
        """

        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            raise FileNotFoundError(self.video_path)
        try:
            bgr = None
            for index in range(self.frame_index + 1):
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise ValueError(
                        f"{self.video_path} ended at frame {index} before frame {self.frame_index}"
                    )
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        finally:
            capture.release()
        if device is not None:
            image = image.to(device)
        return self.load_with_image(image)



class PanopticFrame(LazyCameraDataset[PanopticCameraSpec]):
    """The Panoptic HD cameras at one timestep.

    ``camera_specs`` is one :class:`PanopticCameraSpec` per HD camera.
    """

    def __init__(self, camera_specs: list[PanopticCameraSpec], device: str = "cpu"):
        super().__init__(camera_specs, load_mask=False, load_depth=False, device=device)
