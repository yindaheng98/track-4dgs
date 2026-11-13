"""One Harmony4D sequence and the readers that build its cameras and poses."""

import os
from dataclasses import dataclass

import numpy as np
import torch
from gaussian_splatting.dataset.colmap.read_write_model import qvec2rotmat

from ..dexycb import query_and_tracks_from_trajectories
from ..tracker import Query, Track
from ..utils import fisheye_undistort_grid
from .frame import Harmony4DFrame, HarmonyCameraSpec


@dataclass(frozen=True)
class HarmonyView:
    """One RGB camera and the COLMAP poses registered for it."""

    name: str
    image_dir: str
    pose_dir: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    model: str
    distortion: tuple[float, ...]
    camera_id: int
    poses: dict[int, tuple[torch.Tensor, torch.Tensor]]

    def pose_at(self, frame_index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """World-to-camera pose for ``frame_index``.

        Exo cameras are registered on a few frames and those poses agree, so
        other frames use the nearest one. An ego frame with its own COLMAP
        image uses that pose; otherwise it uses the nearest registered frame.
        """

        if not self.poses:
            raise FileNotFoundError(f"{self.name} has no COLMAP pose")
        if frame_index in self.poses:
            rotation, translation = self.poses[frame_index]
            return rotation, translation, frame_index
        source = min(self.poses, key=lambda index: (abs(index - frame_index), index))
        rotation, translation = self.poses[source]
        return rotation, translation, source


def jpg_count(image_dir: str) -> int:
    stems = sorted(
        int(name[:-4])
        for name in os.listdir(image_dir)
        if name.endswith(".jpg") and name[:-4].isdigit()
    )
    if stems != list(range(1, len(stems) + 1)):
        raise ValueError(f"{image_dir} frame numbers are not 00001.jpg onward")
    return len(stems)


def read_colmap_cameras(path: str) -> dict[int, tuple[str, int, int, tuple[float, ...]]]:
    cameras = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            params = tuple(float(value) for value in parts[4:])
            cameras[int(parts[0])] = (parts[1], int(parts[2]), int(parts[3]), params)
    return cameras


def read_colmap_poses(path: str) -> dict[str, dict[int, tuple[int, torch.Tensor, torch.Tensor]]]:
    grouped: dict[str, dict[int, tuple[int, torch.Tensor, torch.Tensor]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 10 or not parts[-1].endswith(".jpg"):
                continue
            image_name = parts[-1]
            stem = os.path.splitext(os.path.basename(image_name))[0]
            if not stem.isdigit():
                continue
            qvec = np.array([float(value) for value in parts[1:5]], dtype=np.float64)
            tvec = np.array([float(value) for value in parts[5:8]], dtype=np.float64)
            pose = (
                int(parts[-2]),
                torch.tensor(qvec2rotmat(qvec), dtype=torch.float32),
                torch.tensor(tvec, dtype=torch.float32),
            )
            grouped.setdefault(image_name.split("/")[0], {})[int(stem) - 1] = pose
    return grouped


def iter_view_dirs(sequence: str):
    ego_root = os.path.join(sequence, "ego")
    if os.path.isdir(ego_root):
        for name in sorted(os.listdir(ego_root)):
            image_dir = os.path.join(ego_root, name, "images", "rgb")
            pose_dir = os.path.join(sequence, "processed_data", "poses2d", name, "rgb")
            if os.path.isdir(image_dir):
                yield name, image_dir, pose_dir
    exo_root = os.path.join(sequence, "exo")
    for name in sorted(os.listdir(exo_root)):
        image_dir = os.path.join(exo_root, name, "images")
        pose_dir = os.path.join(sequence, "processed_data", "poses2d", name)
        if os.path.isdir(image_dir):
            yield name, image_dir, pose_dir


def load_views(sequence: str) -> tuple[list[HarmonyView], int]:
    workplace = os.path.join(sequence, "colmap", "workplace")
    cameras = read_colmap_cameras(os.path.join(workplace, "cameras.txt"))
    poses = read_colmap_poses(os.path.join(workplace, "images.txt"))
    views = []
    num_frames = None
    for name, image_dir, pose_dir in iter_view_dirs(sequence):
        count = jpg_count(image_dir)
        if num_frames is None:
            num_frames = count
        elif count != num_frames:
            raise ValueError(f"{name} has {count} frames, expected {num_frames}")
        if name not in poses:
            raise FileNotFoundError(f"{name} is missing from {workplace}")
        camera_ids = {camera_id for camera_id, _rotation, _translation in poses[name].values()}
        if len(camera_ids) != 1:
            raise ValueError(f"{name} is registered with more than one COLMAP camera")
        camera_id = next(iter(camera_ids))
        model, width, height, params = cameras[camera_id]
        if len(params) < 4:
            raise ValueError(f"COLMAP camera {camera_id} has no fx, fy, cx, cy")
        view_poses = {
            frame_index: (rotation, translation)
            for frame_index, (_camera_id, rotation, translation) in poses[name].items()
        }
        views.append(HarmonyView(
            name=name,
            image_dir=image_dir,
            pose_dir=pose_dir,
            width=width,
            height=height,
            fx=params[0],
            fy=params[1],
            cx=params[2],
            cy=params[3],
            model=model,
            distortion=params[4:],
            camera_id=camera_id,
            poses=view_poses,
        ))
    if num_frames is None:
        raise FileNotFoundError(f"no cameras under {sequence}")
    return views, num_frames


def read_pose_file(path: str) -> dict[str, np.ndarray]:
    if not os.path.isfile(path):
        return {}
    data = np.load(path, allow_pickle=True)
    item = data.item()
    if not isinstance(item, dict):
        raise ValueError(f"{path} does not store a person-to-joints dict")
    return {str(person): np.asarray(joints, dtype=np.float32)[..., :2] for person, joints in item.items()}


def load_pose_tracks(views: list[HarmonyView], num_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    """SMPL-45 2D joints for every person, concatenated in person-name order."""

    people = set()
    joint_count = None
    for view in views:
        for person, joints in read_pose_file(os.path.join(view.pose_dir, "00001.npy")).items():
            people.add(person)
            if joint_count is None:
                joint_count = int(joints.shape[0])
            elif joints.shape[0] != joint_count:
                raise ValueError(f"{person} joint count {joints.shape[0]} does not match {joint_count}")
    if not people or joint_count is None:
        raise FileNotFoundError("Harmony4D sequence has no 2D pose on frame 1")
    names = sorted(people)
    n_points = len(names) * joint_count
    points = np.full((len(views), num_frames, n_points, 2), -1.0, dtype=np.float32)
    for view_index, view in enumerate(views):
        for frame_index in range(num_frames):
            pose = read_pose_file(os.path.join(view.pose_dir, f"{frame_index + 1:05d}.npy"))
            for person_index, person in enumerate(names):
                if person not in pose:
                    continue
                joints = pose[person]
                if joints.shape[0] != joint_count:
                    raise ValueError(f"{view.name} frame {frame_index + 1} {person} joint count changed")
                start = person_index * joint_count
                points[view_index, frame_index, start:start + joint_count] = joints
    tensor = torch.from_numpy(points)
    valid = torch.isfinite(tensor).all(dim=-1) & (tensor >= 0).all(dim=-1)
    return tensor, valid


@dataclass(frozen=True)
class Harmony4DSpec:
    """One Harmony4D sequence. Cameras and poses are read in :meth:`load`.

    ``sequence`` contains ``ego/``, ``exo/``, ``processed_data/``, and ``colmap/workplace``.
    """

    sequence: str
    start_index: int

    def load(self, device) -> tuple[Query, list[Harmony4DFrame], list[Track]]:
        views, num_frames = load_views(self.sequence)
        if num_frames <= 0:
            raise ValueError("Harmony4D sequence has no frames")
        if self.start_index >= num_frames:
            raise ValueError(f"start_index {self.start_index} is outside 0..{num_frames - 1}")
        camera_specs = [[] for _ in range(num_frames)]
        for view in views:
            grid = fisheye_undistort_grid(
                view.width, view.height,
                view.fx, view.fy,
                view.cx, view.cy,
                view.model, view.distortion,
            )
            for frame_index in range(num_frames):
                rotation, translation, pose_frame = view.pose_at(frame_index)
                camera_specs[frame_index].append(HarmonyCameraSpec(
                    name=view.name,
                    width=view.width,
                    height=view.height,
                    fx=view.fx,
                    fy=view.fy,
                    cx=view.cx,
                    cy=view.cy,
                    rotation=rotation,
                    translation=translation,
                    image_path=os.path.join(view.image_dir, f"{frame_index + 1:05d}.jpg"),
                    camera_id=view.camera_id,
                    grid=grid,
                    pose_frame=pose_frame,
                ))
        frames = [Harmony4DFrame(specs, device=device) for specs in camera_specs]
        points, valid = load_pose_tracks(views, num_frames)
        if device is not None:
            points = points.to(device)
            valid = valid.to(device)
        start_indices = torch.full(
            (points.shape[0], points.shape[2]),
            self.start_index,
            dtype=torch.int64,
            device=points.device,
        )
        query, tracks = query_and_tracks_from_trajectories(
            points,
            valid,
            [view.width for view in views],
            [view.height for view in views],
            start_indices,
        )
        return query, frames, tracks
