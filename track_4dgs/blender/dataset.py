"""Build supervision by rendering stills from a blend file."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import warp as wp
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.utils import focal2fov

from ..dataset import AbstractDataset
from ..dataset.supervision import Supervision
from ..tracker import Query, Track
from .kernels import track_points
from .render_server import RenderServer
from .tracker import BlenderPointTracker
from .workers.render_server_protocol import RenderReply, RenderRequest


@dataclass(frozen=True)
class BlenderScene:
    """One blend file, the query to supervise, and the cameras that film it.

    ``c2w`` is OpenCV camera-to-world in the blend world: the camera looks along
    ``+Z`` and image ``+Y`` points down. Shape is ``[V, D, 4, 4]``, one pose per
    view per frame. ``V`` matches ``query``.
    ``height``, ``width``, and ``focal`` are ``[V, D]``. ``focal`` is pixels,
    ``fx = fy``, and the principal point is the image center.
    ``frames`` is the absolute Blender frame for each column of ``c2w``, length ``D``.
    """

    blend: str
    frames: Sequence[int]
    query: Query
    c2w: torch.Tensor
    height: torch.Tensor
    width: torch.Tensor
    focal: torch.Tensor

    def __post_init__(self):
        c2w = self.c2w
        n_views = self.query.points.shape[0]
        if not torch.is_tensor(c2w) or not torch.is_floating_point(c2w):
            raise TypeError("c2w must be a floating point tensor")
        if c2w.ndim != 4 or c2w.shape[0] != n_views or c2w.shape[-2:] != (4, 4):
            raise ValueError("c2w must be [V, D, 4, 4] with V matching Query")
        if c2w.shape[1] < 1:
            raise ValueError("c2w must contain at least one frame")
        n_frames = c2w.shape[1]
        if len(self.frames) != n_frames:
            raise ValueError("frames must contain one Blender frame per c2w column")
        if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in self.frames):
            raise TypeError("frames must be a sequence of ints")
        shape = (n_views, n_frames)
        for name in ("height", "width", "focal"):
            value = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != shape:
                raise ValueError(f"{name} must be [V, D]")
        if self.height.is_floating_point() or self.height.dtype == torch.bool:
            raise TypeError("height must be an integer tensor")
        if self.width.is_floating_point() or self.width.dtype == torch.bool:
            raise TypeError("width must be an integer tensor")
        if int(self.height.min()) <= 0 or int(self.width.min()) <= 0 or float(self.focal.min()) <= 0:
            raise ValueError("height, width, and focal must be positive")
        frame_indices = self.query.frame_indices
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= n_frames:
            raise ValueError("Query.frame_indices must index c2w frames")

    @property
    def device(self):
        """Device of the query and the camera tensors."""

        return self.query.points.device

    def to(self, device) -> 'BlenderScene':
        """Return this scene with the query and camera tensors on ``device``."""

        return BlenderScene(
            blend=self.blend,
            frames=self.frames,
            query=self.query.to(device),
            c2w=self.c2w.to(device),
            height=self.height.to(device),
            width=self.width.to(device),
            focal=self.focal.to(device),
        )

    def build_render_request(self, view_index: int, frame_index: int) -> RenderRequest:
        """One still: this view and frame, with the pose converted to Blender."""

        c2w = self.c2w[view_index, frame_index].detach().cpu().numpy().astype(np.float64)
        return RenderRequest(
            blend=self.blend,
            frame=self.frames[frame_index],
            width=int(self.width[view_index, frame_index]),
            height=int(self.height[view_index, frame_index]),
            focal=float(self.focal[view_index, frame_index]),
            c2w=c2w @ BLENDER_FROM_OPENCV,
        )

    def build_rendered_camera(self, view_index: int, frame_index: int, reply: RenderReply) -> Camera:
        """One 3DGS camera from this OpenCV pose and one rendered still."""

        c2w = self.c2w[view_index, frame_index]
        height = int(self.height[view_index, frame_index])
        width = int(self.width[view_index, frame_index])
        focal = float(self.focal[view_index, frame_index])
        w2c = torch.linalg.inv(c2w.detach().to(dtype=torch.float32))
        rgb = torch.from_numpy(np.ascontiguousarray(reply.rgb)).permute(2, 0, 1).to(device=self.device, dtype=torch.float32)
        # depth = torch.from_numpy(np.ascontiguousarray(reply.depth)).to(device=device, dtype=torch.float32)
        # object_id = torch.from_numpy(np.ascontiguousarray(reply.object_id)).to(device=device, dtype=torch.float32)
        camera = build_camera(
            image_height=height,
            image_width=width,
            FoVx=focal2fov(focal, width),
            FoVy=focal2fov(focal, height),
            R=w2c[:3, :3].contiguous(),
            T=w2c[:3, 3].contiguous(),
            device=self.device,
        )
        return camera._replace(
            ground_truth_image=rgb,
            # ground_truth_depth=depth,
            # custom_data={"object_id": object_id},
        )
