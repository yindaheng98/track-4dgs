"""Build supervision by rendering stills from a blend file."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.utils import focal2fov

from ..dataset.group_args import AbstractGroupArgsDataset
from ..dataset.supervision import Supervision
from ..tracker import Query
from ..utils import sample_c2ws
from .kernels import sample_near_surface
from .render_server import RenderServer
from .tracker import BlenderPointTracker
from .workers.render_server_protocol import RenderRequest


# OpenCV camera (+Z forward, +Y down) to Blender camera (-Z forward, +Y up).
BLENDER_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True, kw_only=True)
class BlenderScene:
    """One blend file and how to film it.

    Construction records the arguments. :meth:`load_tracker` exports the mesh.
    :meth:`sample_cameras` renders the stills. :meth:`sample_points` draws the query.
    :meth:`load` returns both.
    """

    blend: str
    frame_start: int = 1
    frame_end: Optional[int] = None
    frame_step: Optional[int] = None
    init_dataset_index: int = 0
    n_cameras: int
    height: int
    width: int
    focal: float
    center: tuple[float, float, float]
    radius: tuple[float, float]
    axis: tuple[float, float]
    cap_angle: tuple[float, float]
    spacing: tuple[float, float]
    num_points: int
    visible_cameras: int = 0
    surface_distance: float = 0.0
    device: str = "cuda"

    def __post_init__(self):
        if self.frame_step == 0:
            raise ValueError("frame_step must not be 0")
        if self.height <= 0 or self.width <= 0 or self.focal <= 0:
            raise ValueError("height, width, and focal must be positive")
        if self.init_dataset_index < 0:
            raise ValueError("--init-dataset-index must index one Blender frame")
        if self.frame_end is not None and self.frame_step is not None:
            n_frames = len(range(self.frame_start, self.frame_end + 1, self.frame_step))
            if self.init_dataset_index >= n_frames:
                raise ValueError("--init-dataset-index must index one Blender frame")

    def load_tracker(self, blender: str) -> BlenderPointTracker:
        """Export this blend's rigid meshes."""

        return BlenderPointTracker(
            self.blend, blender=blender,
            frame_start=self.frame_start, frame_end=self.frame_end, frame_step=self.frame_step,
        ).to(self.device)

    def camera(self, c2w: torch.Tensor) -> Camera:
        """One 3DGS camera for this OpenCV pose. The image is filled in later."""

        w2c = torch.linalg.inv(c2w.detach().to(dtype=torch.float32))
        return build_camera(
            image_height=self.height,
            image_width=self.width,
            FoVx=focal2fov(self.focal, self.width),
            FoVy=focal2fov(self.focal, self.height),
            R=w2c[:3, :3].contiguous(),
            T=w2c[:3, 3].contiguous(),
            device=self.device,
        )

    def render_camera(self, render_server: RenderServer, camera: Camera, c2w: torch.Tensor, frame: int) -> Camera:
        """Render one still and attach it to ``camera``."""

        pose = c2w.detach().cpu().numpy().astype(np.float64) @ BLENDER_FROM_OPENCV
        reply = render_server.render(RenderRequest(
            blend=self.blend, frame=frame, width=self.width, height=self.height, focal=float(self.focal), c2w=pose,
        ))
        rgb = torch.from_numpy(np.ascontiguousarray(reply.rgb)).permute(2, 0, 1).to(device=self.device, dtype=torch.float32)
        return camera._replace(ground_truth_image=rgb)

    def sample_cameras(self, tracker: BlenderPointTracker, render_server: RenderServer) -> tuple[list[Camera], list[list[Camera]]]:
        """Sample the views and render one still per view and frame.

        Returns the cameras and one still per view for each exported frame.
        The view count may be smaller than ``n_cameras``.
        """

        n_frames = int(tracker.matrices.shape[0])
        if n_frames < 1:
            raise ValueError("the blend scene produced no frames")
        frame_numbers = range(tracker.frame_start, tracker.frame_end + 1, tracker.frame_step)
        if len(frame_numbers) != n_frames:
            raise ValueError(f"requested {len(frame_numbers)} Blender frames but the blend exported {n_frames}")
        c2ws = sample_c2ws(
            self.n_cameras, self.center, self.radius, self.axis,
            self.cap_angle, self.spacing, self.device)
        cameras = [self.camera(c2ws[view_index]) for view_index in range(int(c2ws.shape[0]))]
        frames = [
            [self.render_camera(render_server, cameras[view_index], c2ws[view_index], frame) for view_index in range(len(cameras))]
            for frame in frame_numbers
        ]
        return cameras, frames

    def sample_points(self, tracker: BlenderPointTracker, cameras: list[Camera]) -> Query:
        """Sample a query on the init frame of ``tracker`` that ``cameras`` can see."""

        n_frames = int(tracker.matrices.shape[0])
        if self.init_dataset_index >= n_frames:
            raise ValueError("--init-dataset-index must index one Blender frame")
        xyz = sample_near_surface(
            tracker.verts, tracker.faces, tracker.matrices[self.init_dataset_index], cameras,
            self.num_points, self.visible_cameras, self.surface_distance)
        frame_indices = torch.full((xyz.shape[0],), self.init_dataset_index, dtype=torch.long, device=xyz.device)
        return Query.from_projection(xyz, [cameras] * n_frames, frame_indices)

    def load(self, tracker: BlenderPointTracker, render_server: RenderServer) -> tuple[Query, list[list[Camera]]]:
        """Sample a query on ``tracker`` and render one still per view and frame."""

        with torch.no_grad():
            cameras, frames = self.sample_cameras(tracker, render_server)
            return self.sample_points(tracker, cameras), frames


class BlenderDataset(AbstractGroupArgsDataset):
    """One deferred blend per ``--blend``.

    :meth:`from_args` records specs. :meth:`__getitem__` exports that blend,
    renders its stills, and tracks the query with the same mesh.
    """

    arg_group_starters = ("--blend",)

    def __init__(self, *scenes: BlenderScene, blender: str = "blender"):
        self.scenes = scenes
        self.blender = blender
        self.device = None
        self.render_server = RenderServer(blender)

    @staticmethod
    def add_group_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register one ``--blend`` group's flags on ``parser`` and return ``parser``."""
        parser.add_argument("--blend", required=True, type=str, metavar="FILE", help="Blend file for one sample.")
        parser.add_argument("--frame_start", type=int, metavar="START", default=1, help="First absolute Blender frame for this blend. Defaults to 1.")
        parser.add_argument("--frame_end", type=int, metavar="END", default=None, help="Last absolute Blender frame for this blend. Defaults to the blend scene.")
        parser.add_argument("--frame_step", type=int, metavar="STEP", default=None, help="Frame step for this blend. Defaults to the blend scene.")
        parser.add_argument("--init-dataset-index", type=int, metavar="INDEX", default=0, help="Index into range(frame_start, frame_end + 1, frame_step) for the mesh the query is sampled on. Defaults to 0.")
        return parser

    @staticmethod
    def add_common_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register the cameras sampled around every blend and return ``parser``."""
        parser.add_argument("--n-cameras", required=True, type=int)
        parser.add_argument("--height", required=True, type=int)
        parser.add_argument("--width", required=True, type=int)
        parser.add_argument("--focal", required=True, type=float, help="Focal length in pixels. fx = fy, principal point at the image center.")
        parser.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"), help="Shell center and look-at point.")
        parser.add_argument("--radius", type=float, nargs=2, default=(10.0, 12.0), metavar=("MIN", "MAX"))
        parser.add_argument("--axis", type=float, nargs=2, default=(0.0, 90.0), metavar=("AZIMUTH", "ELEVATION"), help="Cap axis direction in degrees. Azimuth 0 points along +X; elevation 90 points straight up.")
        parser.add_argument("--cap-angle", type=float, nargs=2, default=(0.0, 90.0), metavar=("MIN", "MAX"), help="Band edges in degrees from the cap axis. 0 90 is the upper hemisphere when elevation is 90.")
        parser.add_argument("--spacing", type=float, nargs=2, default=(10.0, 45.0), metavar=("MIN", "MAX"), help="Angle in degrees between a camera and its nearest neighbor.")
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--num-points", default=256, type=int)
        parser.add_argument("--visible-cameras", default=0, type=int, help="Keep a point only when more cameras than this see it.")
        parser.add_argument("--surface-distance", default=0.0, type=float, help="Half-width of the band around the mesh, in world units.")
        parser.add_argument("--blender", default="blender", type=str)
        return parser

    @staticmethod
    def spec_from_args(common_args: Namespace, group_args: Namespace) -> BlenderScene:
        """One :class:`BlenderScene` for this ``--blend`` group.

        Does not open the blend. :meth:`BlenderScene.load` does that.
        """
        return BlenderScene(
            blend=group_args.blend,
            frame_start=group_args.frame_start,
            frame_end=group_args.frame_end,
            frame_step=group_args.frame_step,
            init_dataset_index=group_args.init_dataset_index,
            n_cameras=common_args.n_cameras,
            height=common_args.height,
            width=common_args.width,
            focal=common_args.focal,
            center=tuple(common_args.center),
            radius=tuple(common_args.radius),
            axis=tuple(common_args.axis),
            cap_angle=tuple(common_args.cap_angle),
            spacing=tuple(common_args.spacing),
            num_points=common_args.num_points,
            visible_cameras=common_args.visible_cameras,
            surface_distance=common_args.surface_distance,
            device=common_args.device,
        )

    @classmethod
    def parse_known_group_chunk(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse one blend file and its frame range.

        Frames are ``range(frame_start, frame_end + 1, frame_step)``.
        ``frame_start`` defaults to 1. ``frame_end`` and ``frame_step`` default to the blend scene.
        ``--init-dataset-index`` defaults to 0.
        """
        parser = ArgumentParser(allow_abbrev=False)
        cls.add_group_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def parse_known_common_argv(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse the cameras sampled around each blend."""
        parser = ArgumentParser(allow_abbrev=False)
        cls.add_common_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def from_args(cls, common_args: Namespace, *group_args: Namespace) -> 'BlenderDataset':
        """One deferred blend per ``--blend``. The mesh is exported in :meth:`__getitem__`."""
        assert group_args, "at least one --blend is required"
        scenes = tuple(cls.spec_from_args(common_args, group) for group in group_args)
        return cls(*scenes, blender=common_args.blender)

    def __len__(self) -> int:
        return len(self.scenes)

    def to(self, device) -> 'BlenderDataset':
        """Export and render later samples on ``device``."""
        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        scene = self.scenes[index]
        if self.device is not None:
            scene = replace(scene, device=self.device)
        tracker = scene.load_tracker(self.blender)
        query, frames = scene.load(tracker, self.render_server)
        tracks = tracker(query, frames)
        return Supervision(query=query, frames=frames, tracks=tracks)
