"""Panoptic HD views decoded from each camera video.

:class:`PanopticFrame` is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`
of :class:`PanopticCameraSpec`. :class:`PanopticFrames` is used as
``list[PanopticFrame]``. A frame with :attr:`PanopticFrame.frames` set asks that
group, on a cache miss, to decode the view once and fill every frame.
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


class PanopticFrames:
    """The selected timesteps, and the link between their :class:`PanopticFrame` objects.

    ``frames`` is a list of :class:`PanopticFrame`. Each one is unlinked:
    ``PanopticFrame.frames`` is ``None``. This constructor stores that list and
    then sets ``frame.frames`` to this group. ``len`` and iteration follow the
    list. :meth:`load_view` is what a frame calls when its cache has no camera
    for a view: one sequential read of that video, then each frame's cache
    receives its camera.
    """

    def __init__(self, frames: list["PanopticFrame"]):
        if not frames:
            raise ValueError("no frames")
        if any(frame.frames is not None for frame in frames):
            raise ValueError("PanopticFrame.frames must be None")
        n_views = len(frames[0])
        if n_views == 0 or any(len(frame) != n_views for frame in frames):
            raise ValueError("every frame must list the same cameras")
        self.device = frames[0].device
        self.frames = list(frames)
        for frame in self.frames:
            frame.frames = self

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> "PanopticFrame":
        return self.frames[index]

    def __iter__(self):
        return iter(self.frames)

    def load_view(self, view_index: int) -> None:
        """Decode one view and store each selected frame on that frame's cache.

        Every frame in this view shares one video. Decoding starts at frame 0
        and stops at the last selected index. Each selected frame becomes
        ``(3, H, W)`` float32 RGB in ``[0, 1]`` on :attr:`device`.
        """

        specs = [frame.camera_specs[view_index] for frame in self.frames]
        video_path = specs[0].video_path
        if any(spec.video_path != video_path for spec in specs):
            raise ValueError(f"{specs[0].name} frames do not share {video_path}")
        position_at_frame: dict[int, int] = {}
        for position, spec in enumerate(specs):
            if spec.frame_index in position_at_frame:
                raise ValueError(f"{video_path} lists frame {spec.frame_index} twice")
            position_at_frame[spec.frame_index] = position
        last_frame = max(position_at_frame)
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(video_path)
        try:
            for index in range(last_frame + 1):
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise ValueError(f"{video_path} ended at frame {index} before frame {last_frame}")
                position = position_at_frame.get(index)
                if position is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
                if self.device is not None:
                    image = image.to(self.device)
                self.frames[position]._cache[view_index] = specs[position].load_with_image(image)
        finally:
            capture.release()

    def to(self, device) -> None:
        """Read later cameras on ``device``. A change drops every frame's ``_cache``."""

        if self.device == device:
            return
        self.device = device
        for frame in self.frames:
            frame.device = device
            frame._cache.clear()
            frame._cache_state = (frame.load_mask, frame.load_depth, device)


class PanopticFrame(LazyCameraDataset[PanopticCameraSpec]):
    """The HD cameras at one timestep.

    ``camera_specs`` is one :class:`PanopticCameraSpec` per HD camera.
    ``len`` counts cameras. Cameras already read stay in
    :attr:`~gaussian_splatting.dataset.LazyCameraDataset._cache`.
    ``frames`` is ``None`` until a :class:`PanopticFrames` links this frame.
    Without it, indexing calls :meth:`PanopticCameraSpec.load`. With it, a cache
    miss calls :meth:`PanopticFrames.load_view`, which stores that view on every
    connected frame, and the camera is then read from ``_cache``.
    """

    frames: "PanopticFrames | None" = None

    def __init__(self, camera_specs: list[PanopticCameraSpec], device: str = "cpu"):
        super().__init__(camera_specs, load_mask=False, load_depth=False, device=device)
        self.frames = None

    def __getitem__(self, idx: int) -> Camera:
        if self.frames is not None:
            state = (self.load_mask, self.load_depth, self.device)
            if state != self._cache_state:
                self._cache.clear()
                self._cache_state = state
            if idx not in self._cache:
                self.frames.load_view(idx)
        return super().__getitem__(idx)

    def to(self, device) -> "PanopticFrame":
        """Read later cameras on ``device``. A linked group drops every frame's ``_cache``."""

        if self.frames is not None:
            self.frames.to(device)
        return super().to(device)
