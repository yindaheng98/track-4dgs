"""Build supervision by rendering stills from a blend file."""

from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.utils import focal2fov

from ..dataset import AbstractArgumentParserDataset
from ..dataset.supervision import Supervision
from ..tracker import Query, Track
from ..utils import sample_c2ws
from .kernels import sample_near_surface, track_points
from .render_server import RenderServer
from .tracker import BlenderPointTracker
from .workers.render_server_protocol import RenderReply, RenderRequest


# OpenCV camera (+Z forward, +Y down) to Blender camera (-Z forward, +Y up).
BLENDER_FROM_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)


@dataclass(frozen=True)
class BlenderScene:
    """One blend file, the query to supervise, and the cameras that film it.

    ``c2w`` is OpenCV camera-to-world in the blend world: the camera looks along
    ``+Z`` and image ``+Y`` points down. Shape is ``[V, D, 4, 4]``, one pose per
    view per frame. ``V`` matches ``query``.
    ``height``, ``width``, and ``focal`` are ``[V, D]``. ``focal`` is pixels,
    ``fx = fy``, and the principal point is the image center.
    Columns of ``c2w`` are ``range(frame_start, frame_end + 1, frame_step)``.
    """

    blend: str
    frame_start: int
    frame_end: int
    frame_step: int
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
        if not all(isinstance(value, int) for value in (self.frame_start, self.frame_end, self.frame_step)):
            raise TypeError("frame_start, frame_end, and frame_step must be ints")
        if self.frame_step == 0:
            raise ValueError("frame_step must not be 0")
        if len(self.frames) != n_frames:
            raise ValueError("c2w columns must match range(frame_start, frame_end + 1, frame_step)")
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

    @property
    def frames(self) -> range:
        """Absolute Blender frames, one per column of ``c2w``."""

        return range(self.frame_start, self.frame_end + 1, self.frame_step)

    def to(self, device) -> 'BlenderScene':
        """Return this scene with the query and camera tensors on ``device``."""

        return BlenderScene(
            blend=self.blend,
            frame_start=self.frame_start,
            frame_end=self.frame_end,
            frame_step=self.frame_step,
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


class AppendBlenderScene(Action):
    """Attach frame bounds and ``--init-dataset-index`` to the preceding ``--blend``.

    Each ``--blend`` starts one scene. ``--frame_start``, ``--frame_end``,
    ``--frame_step``, and ``--init-dataset-index`` may each follow it once.
    A second copy, or any of them before the first ``--blend``, is an error.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        scenes = getattr(namespace, "scenes", None)
        if scenes is None:
            scenes = []
            namespace.scenes = scenes
        if option_string == "--blend":
            scenes.append(Namespace(
                blend=values, frame_start=None, frame_end=None, frame_step=None, init_dataset_index=None))
            return
        if not scenes:
            parser.error(f"{option_string} must follow --blend")
        scene = scenes[-1]
        field = {
            "--frame_start": "frame_start",
            "--frame_end": "frame_end",
            "--frame_step": "frame_step",
            "--init-dataset-index": "init_dataset_index",
        }[option_string]
        if getattr(scene, field) is not None:
            parser.error(f"{option_string} already set for --blend {scene.blend}")
        setattr(scene, field, values)


class BlenderDataset(AbstractArgumentParserDataset):
    """Render each :class:`BlenderScene` into one :class:`Supervision` sample.

    Images and depth come from :class:`RenderServer`.
    Tracks bind the query to the blend's rigid meshes.
    """

    def __init__(self, *scenes: BlenderScene, blender: str = "blender"):
        self.scenes = scenes
        self.blender = blender
        self.render_server = RenderServer(blender)

    @classmethod
    def add_argument(cls, parser: ArgumentParser) -> None:
        """Register one blend file, its frame range, and the cameras sampled around it.

        ``--frame_start``, ``--frame_end``, ``--frame_step``, and ``--init-dataset-index``
        belong to the preceding ``--blend``. Each may appear once in that group.
        Frames are ``range(frame_start, frame_end + 1, frame_step)``.
        ``frame_start`` defaults to 1. ``frame_end`` and ``frame_step`` default to the blend scene.
        ``--init-dataset-index`` defaults to 0.
        """
        parser.add_argument("--blend", required=True, dest="scenes", action=AppendBlenderScene, type=str, metavar="FILE", help="Blend file for one sample. Repeat to start another sample.")
        parser.add_argument("--frame_start", dest="scenes", action=AppendBlenderScene, type=int, metavar="START", help="First absolute Blender frame for the preceding --blend. At most once. Defaults to 1.")
        parser.add_argument("--frame_end", dest="scenes", action=AppendBlenderScene, type=int, metavar="END", help="Last absolute Blender frame for the preceding --blend. At most once. Defaults to the blend scene.")
        parser.add_argument("--frame_step", dest="scenes", action=AppendBlenderScene, type=int, metavar="STEP", help="Frame step for the preceding --blend. At most once. Defaults to the blend scene.")
        parser.add_argument("--init-dataset-index", dest="scenes", action=AppendBlenderScene, type=int, metavar="INDEX", help="Index into range(frame_start, frame_end + 1, frame_step) for the mesh the query is sampled on. At most once. Defaults to 0.")
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

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'BlenderDataset':
        """Sample a query near each blend's surface and film it with cameras on a spherical cap."""
        assert args.scenes, "at least one --blend is required"
        scenes = []
        with torch.no_grad():
            for spec in args.scenes:
                blend = spec.blend
                frame_start = 1 if spec.frame_start is None else spec.frame_start
                init_dataset_index = 0 if spec.init_dataset_index is None else spec.init_dataset_index
                mesh = BlenderPointTracker(
                    blend, blender=args.blender,
                    frame_start=frame_start, frame_end=spec.frame_end, frame_step=spec.frame_step,
                ).to(args.device)
                n_frames = mesh.matrices.shape[0]
                assert n_frames, "the blend scene produced no frames."
                assert 0 <= init_dataset_index < n_frames, "--init-dataset-index must index one Blender frame."
                query_matrix = mesh.matrices[init_dataset_index]
                c2ws = sample_c2ws(
                    args.n_cameras, args.center, args.radius, args.axis,
                    args.cap_angle, args.spacing, args.device)
                cameras: list[Camera] = []
                for c2w in c2ws:
                    w2c = torch.linalg.inv(c2w)
                    cameras.append(build_camera(
                        image_height=args.height,
                        image_width=args.width,
                        FoVx=focal2fov(args.focal, args.width),
                        FoVy=focal2fov(args.focal, args.height),
                        R=w2c[:3, :3].contiguous(),
                        T=w2c[:3, 3].contiguous(),
                        device=c2w.device,
                    ))
                xyz = sample_near_surface(
                    mesh.verts, mesh.faces, query_matrix, cameras,
                    args.num_points, args.visible_cameras, args.surface_distance)
                frame_indices = torch.full((args.num_points,), init_dataset_index, dtype=torch.long, device=args.device)
                query = Query.from_projection(xyz, [cameras] * n_frames, frame_indices)
                n_views = c2ws.shape[0]
                scenes.append(BlenderScene(
                    blend=blend, frame_start=mesh.frame_start, frame_end=mesh.frame_end, frame_step=mesh.frame_step, query=query,
                    c2w=c2ws.unsqueeze(1).expand(-1, n_frames, -1, -1).contiguous(),
                    height=torch.full((n_views, n_frames), args.height, dtype=torch.long, device=args.device),
                    width=torch.full((n_views, n_frames), args.width, dtype=torch.long, device=args.device),
                    focal=torch.full((n_views, n_frames), args.focal, dtype=torch.float32, device=args.device),
                ))
        return cls(*scenes, blender=args.blender)

    def __len__(self) -> int:
        return len(self.scenes)

    def to(self, device) -> 'BlenderDataset':
        self.scenes = tuple(scene.to(device) for scene in self.scenes)
        return self

    def __getitem__(self, index: int) -> Supervision:
        scene = self.scenes[index]
        frames = [self.render_frame(scene, frame_index) for frame_index in range(scene.c2w.shape[1])]
        return Supervision(query=scene.query, frames=frames, tracks=self.tracks(scene, frames))

    def render_frame(self, scene: BlenderScene, frame_index: int) -> list[Camera]:
        """Render every view of one frame."""

        cameras = []
        for view_index in range(scene.c2w.shape[0]):
            request = scene.build_render_request(view_index, frame_index)
            reply = self.render_server.render(request)
            camera = scene.build_rendered_camera(view_index, frame_index, reply)
            cameras.append(camera)
        return cameras

    def tracks(self, scene: BlenderScene, frames: Sequence[Sequence[Camera]]) -> list[Track]:
        """Bind ``scene.query`` to the rigid meshes posed across its frame range."""

        query = scene.query
        tracker = BlenderPointTracker(
            scene.blend, blender=self.blender,
            frame_start=scene.frame_start, frame_end=scene.frame_end, frame_step=scene.frame_step,
        ).to(scene.device)
        n_views, n_points, _ = query.points.shape
        n_frames = len(frames)
        if tracker.matrices.shape[0] != n_frames:
            raise ValueError(
                f"requested {n_frames} Blender frames but the blend exported {tracker.matrices.shape[0]}"
            )
        points, visibility, valid = track_points(
            tracker.verts, tracker.faces, tracker.matrices, scene.query, frames, tracker.eps, str(tracker.device),
        )
        confidence = query.points.new_ones((n_views, n_frames, n_points))
        mask = valid.to(dtype=torch.bool).expand(n_frames, n_points)
        return [
            Track(
                points=points[view_index],
                visibility=visibility[view_index],
                confidence=confidence[view_index],
                mask=mask,
            )
            for view_index in range(n_views)
        ]
