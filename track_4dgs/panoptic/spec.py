"""One Panoptic sequence and the readers that build its HD cameras and body joints."""

import json
import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
from ..dexycb.trajectory import query_and_tracks_from_trajectories
from ..tracker import Query, Track
from ..utils import full_proj_transform, opencv_undistort_grid, project, selected_frame_indices
from .body import sample_body_surface
from .frame import PanopticCameraSpec, PanopticFrame, PanopticFrames

BODY_JOINT_COUNT = 19
BODY_JOINT_NAMES = (
    "Neck",
    "Nose",
    "BodyCenter",
    "LShoulder",
    "LElbow",
    "LWrist",
    "LHip",
    "LKnee",
    "LAnkle",
    "RShoulder",
    "RElbow",
    "RWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LEye",
    "LEar",
    "REye",
    "REar",
)
CENTIMETERS_TO_METERS = 0.01


@dataclass(frozen=True)
class PanopticCamera:
    """One HD camera. Translation is meters. Distortion is the OpenCV coefficient vector."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]
    rotation: torch.Tensor
    translation: torch.Tensor


def scale_camera(camera: PanopticCamera, max_image_size: Optional[int] = None) -> PanopticCamera:
    """Shrink ``camera`` until its long side is at most ``max_image_size``.

    ``build_camera`` resizes with ``align_corners=False``, which sends source
    pixel ``u`` to ``(u + 0.5) * new / old - 0.5``. Focal lengths scale by
    ``new / old``. The principal point uses that same mapping. ``None``, and a
    camera that is already small enough, return the same camera. Each side
    stays at least 2.
    """

    if max_image_size is None or max(camera.width, camera.height) <= max_image_size:
        return camera
    scale = max_image_size / max(camera.width, camera.height)
    width = max(2, int(round(camera.width * scale)))
    height = max(2, int(round(camera.height * scale)))
    sx = width / camera.width
    sy = height / camera.height
    return replace(
        camera,
        width=width,
        height=height,
        fx=camera.fx * sx,
        fy=camera.fy * sy,
        cx=(camera.cx + 0.5) * sx - 0.5,
        cy=(camera.cy + 0.5) * sy - 0.5,
    )


def load_hd_cameras(sequence: str) -> list[PanopticCamera]:
    """Every HD camera in ``calibration_<sequence>.json``, sorted by name.

    VGA and Kinect entries are omitted: the body jsons are indexed at the HD frame rate.
    """

    name = os.path.basename(os.path.normpath(sequence))
    with open(os.path.join(sequence, f"calibration_{name}.json"), encoding="utf-8") as handle:
        payload = json.load(handle)
    selected = []
    seen = set()
    for camera in payload["cameras"]:
        if camera["type"] != "hd":
            continue
        name = str(camera["name"])
        if name in seen:
            raise ValueError(f"{sequence} lists HD camera {name} twice")
        seen.add(name)
        selected.append(camera)
    if not selected:
        raise ValueError(f"{sequence} has no HD cameras")
    selected.sort(key=lambda camera: str(camera["name"]))
    cameras = []
    for camera in selected:
        matrix = np.asarray(camera["K"], dtype=np.float64)
        rotation = np.asarray(camera["R"], dtype=np.float64)
        translation = np.asarray(camera["t"], dtype=np.float64).reshape(3)
        width, height = (int(value) for value in camera["resolution"])
        if matrix.shape != (3, 3) or rotation.shape != (3, 3):
            raise ValueError(f"{camera['name']} calibration is not 3x3")
        cameras.append(PanopticCamera(
            name=str(camera["name"]),
            width=width,
            height=height,
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
            distortion=tuple(float(value) for value in camera["distCoef"]),
            rotation=torch.tensor(rotation, dtype=torch.float32),
            translation=torch.tensor(translation * CENTIMETERS_TO_METERS, dtype=torch.float32),
        ))
    return cameras


def pose_frame_count(folder: str) -> int:
    """Frames ``0 .. N-1`` from ``body3DScene_XXXXXXXX.json``."""

    indices = []
    prefix = "body3DScene_"
    for name in os.listdir(folder):
        if not (name.startswith(prefix) and name.endswith(".json")):
            continue
        stem = name[len(prefix):-len(".json")]
        if stem.isdigit():
            indices.append(int(stem))
    indices.sort()
    if indices != list(range(len(indices))):
        raise ValueError(f"{folder} body jsons are not 00000000.json onward without gaps")
    if not indices:
        raise ValueError(f"{folder} has no body json")
    return len(indices)


def read_bodies(path: str) -> dict[int, torch.Tensor]:
    """One frame: body id to ``(19, 4)`` joints.

    The first three columns are meters. The fourth is the dataset confidence.
    """

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    bodies = {}
    for body in payload["bodies"]:
        person_id = int(body["id"])
        joints = torch.tensor(body["joints19"], dtype=torch.float32).reshape(-1, 4)
        if joints.shape != (BODY_JOINT_COUNT, 4):
            raise ValueError(
                f"{path} body {person_id} has shape {tuple(joints.shape)}, expected ({BODY_JOINT_COUNT}, 4)"
            )
        if person_id in bodies:
            raise ValueError(f"{path} lists body {person_id} twice")
        joints = joints.clone()
        joints[:, :3] *= CENTIMETERS_TO_METERS
        bodies[person_id] = joints
    return bodies


def load_body_tracks(
        cameras: list[PanopticCamera],
        num_frames: int,
        pose_dir: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project the COCO-19 joints into every HD camera.

    People are concatenated in id order, so point ``person * 19 + joint`` is
    the same joint in every view. A joint is invalid when its person is
    absent, its confidence is not positive, it is behind the camera, or the
    pixel leaves the image. Pixels follow :func:`~track_4dgs.utils.project`,
    which is the OpenCV pixel minus one half.
    Returns pixels ``(V, F, N, 2)``, a bool mask ``(V, F, N)``, and the joint
    positions ``(F, N, 3)`` in meters.
    """

    frames = [
        read_bodies(os.path.join(pose_dir, f"body3DScene_{frame_index:08d}.json"))
        for frame_index in range(num_frames)
    ]
    person_ids = sorted({person_id for frame in frames for person_id in frame})
    if not person_ids:
        raise FileNotFoundError(f"{pose_dir} has no bodies")
    n_points = len(person_ids) * BODY_JOINT_COUNT
    xyz = torch.zeros((num_frames, n_points, 3))
    present = torch.zeros((num_frames, n_points), dtype=torch.bool)
    for frame_index, bodies in enumerate(frames):
        for person_index, person_id in enumerate(person_ids):
            joints = bodies.get(person_id)
            if joints is None:
                continue
            slot = slice(person_index * BODY_JOINT_COUNT, (person_index + 1) * BODY_JOINT_COUNT)
            xyz[frame_index, slot] = joints[:, :3]
            present[frame_index, slot] = joints[:, 3] > 0
    points = torch.full((len(cameras), num_frames, n_points, 2), -1.0)
    valid = torch.zeros((len(cameras), num_frames, n_points), dtype=torch.bool)
    for view_index, camera in enumerate(cameras):
        full_proj = full_proj_transform(
            camera.fx, camera.fy, camera.cx, camera.cy,
            camera.width, camera.height, camera.rotation, camera.translation,
        )
        for frame_index in range(num_frames):
            uv, in_frustum = project(xyz[frame_index], full_proj, camera.width, camera.height)
            seen = present[frame_index]
            points[view_index, frame_index, seen] = uv[seen]
            valid[view_index, frame_index, seen] = in_frustum[seen]
    return points, valid, xyz


@dataclass(frozen=True)
class PanopticSpec:
    """One Panoptic sequence. HD cameras and body joints are read in :meth:`load`.

    ``sequence`` contains ``calibration_<name>.json``,
    ``hdPose3d_stage1_coco19/body3DScene_XXXXXXXX.json``, and one HD video per
    camera at ``videos/hd_shared_crf20/hd_<camera>.mp4``.
    Frame 0 is the first frame of that video.
    Joints are stored in centimeters and converted to meters.
    ``num_points`` is the total number of tracks and defaults to 256.
    The 19 body joints of each person come first. The rest are samples on a
    ribbon 4 cm beside each bone, drawn with the same surface sampler as
    Harmony4D. A sample is kept when more cameras than ``visible_cameras`` see
    it, then carried with that triangle. ``num_points`` must cover every body
    joint. ``inward_distance`` and ``outward_distance`` move a sample along the
    ribbon normal, in meters. ``occlusion_eps`` is the occlusion tolerance in
    meters.
    Frames are ``range(frame_start, frame_end + 1, frame_step)`` using the HD
    frame index. ``frame_end`` defaults to the last body json. ``frame_step``
    defaults to 1. ``init_dataset_index`` indexes that range and is the query.
    ``max_image_size`` caps the long side of each image. A larger photo is
    scaled down with its intrinsics. ``None`` keeps the calibration resolution.
    """

    sequence: str
    frame_start: int = 0
    frame_end: Optional[int] = None
    frame_step: int = 1
    init_dataset_index: int = 0
    num_points: int = 256
    visible_cameras: int = 2
    inward_distance: float = 0.0
    outward_distance: float = 0.0
    occlusion_eps: float = 1e-2
    max_image_size: Optional[int] = None

    def __post_init__(self):
        if self.frame_step == 0:
            raise ValueError("frame_step must not be 0")
        if self.num_points < 0:
            raise ValueError("num_points must be non-negative")
        if self.inward_distance < 0 or self.outward_distance < 0:
            raise ValueError("inward_distance and outward_distance must be non-negative")
        if self.occlusion_eps < 0:
            raise ValueError("occlusion_eps must be non-negative")
        if self.init_dataset_index < 0:
            raise ValueError("--init-dataset-index must index one frame")
        if self.max_image_size is not None and self.max_image_size < 2:
            raise ValueError("max_image_size must be at least 2")
        if self.frame_end is not None:
            n_frames = len(range(self.frame_start, self.frame_end + 1, self.frame_step))
            if self.init_dataset_index >= n_frames:
                raise ValueError("--init-dataset-index must index one frame")

    def load(self, device) -> tuple[Query, PanopticFrames, list[Track]]:
        cameras = [scale_camera(camera, self.max_image_size) for camera in load_hd_cameras(self.sequence)]
        poses = os.path.join(self.sequence, "hdPose3d_stage1_coco19")
        num_frames = pose_frame_count(poses)
        if num_frames <= 0:
            raise ValueError(f"{self.sequence} has no frames")
        indices = selected_frame_indices(self.frame_start, self.frame_end, self.frame_step, num_frames)
        if self.init_dataset_index >= len(indices):
            raise ValueError("--init-dataset-index must index one frame")
        grids = [
            opencv_undistort_grid(
                camera.width, camera.height, camera.fx, camera.fy, camera.cx, camera.cy, camera.distortion,
            )
            for camera in cameras
        ]
        panoptic_frames = []
        for frame_index in indices:
            specs = []
            for camera, grid in zip(cameras, grids, strict=True):
                specs.append(PanopticCameraSpec(
                    name=camera.name,
                    width=camera.width,
                    height=camera.height,
                    fx=camera.fx,
                    fy=camera.fy,
                    cx=camera.cx,
                    cy=camera.cy,
                    rotation=camera.rotation,
                    translation=camera.translation,
                    grid=grid,
                    video_path=os.path.join(
                        self.sequence, "videos", "hd_shared_crf20", f"hd_{camera.name}.mp4",
                    ),
                    frame_index=frame_index,
                ))
            panoptic_frames.append(PanopticFrame(specs, device=device))
        frames = PanopticFrames(panoptic_frames)
        points, valid, xyz = load_body_tracks(cameras, num_frames, poses)
        points = points[:, indices]
        valid = valid[:, indices]
        joint_count = points.shape[2]
        if self.num_points < joint_count:
            raise ValueError(f"num_points is the total and must cover the {joint_count} body joints")
        if self.num_points > joint_count:
            people = joint_count // BODY_JOINT_COUNT
            surface_points, surface_valid = sample_body_surface(
                xyz[indices].view(len(indices), people, BODY_JOINT_COUNT, 3),
                frames,
                self.init_dataset_index,
                self.num_points - joint_count,
                self.visible_cameras,
                self.inward_distance,
                self.outward_distance,
                self.occlusion_eps,
                device,
            )
            points = torch.cat([points.to(surface_points.device), surface_points], dim=2)
            valid = torch.cat([valid.to(surface_valid.device), surface_valid], dim=2)
        elif device is not None:
            points = points.to(device)
            valid = valid.to(device)
        start_indices = torch.full(
            (points.shape[0], points.shape[2]),
            self.init_dataset_index,
            dtype=torch.int64,
            device=points.device,
        )
        query, tracks = query_and_tracks_from_trajectories(
            points,
            valid,
            [camera.width for camera in cameras],
            [camera.height for camera in cameras],
            start_indices,
        )
        return query, frames, tracks
