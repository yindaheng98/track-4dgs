import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import tifffile
from PIL import Image
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.dataset.colmap.dataset import ColmapCameraDataset
from gaussian_splatting.dataset.colmap.read_write_model import Camera as ColmapCamera, Image as ColmapImage, rotmat2qvec, write_model
from gaussian_splatting.utils import fov2focal

from ..tracker import Query, Track


@dataclass(frozen=True)
class Supervision:
    """One point-tracking sample: query, frames, and per-view tracks.

    Field layout matches :meth:`track_4dgs.tracker.tracker.AbstractPointTracker.__call__`.
    ``query`` is view-major with one row per camera/view. ``frames`` is
    frame-major: one :class:`~gaussian_splatting.dataset.CameraDataset` per
    frame, with cameras in the same order as ``query``. ``tracks`` is one
    :class:`~track_4dgs.tracker.tracker.Track` per view. Each track has
    ``points`` shaped ``[D, N, 2]`` and ``visibility`` / ``confidence`` /
    ``mask`` shaped ``[D, N]``, where ``D == len(frames)`` and ``N`` is the
    query point count.
    """

    query: Query
    frames: Sequence[CameraDataset]
    tracks: Sequence[Track]

    def __post_init__(self):
        if len(self.frames) == 0:
            raise ValueError("Supervision.frames must not be empty")
        n_views = len(self.frames[0])
        if any(len(dataset) != n_views for dataset in self.frames):
            raise ValueError("frames must all contain the same number of cameras")
        if self.query.points.shape[0] != n_views:
            raise ValueError("Query must have one row per camera/view")
        if len(self.tracks) != n_views:
            raise ValueError("tracks must contain one Track per view")
        n_frames = len(self.frames)
        n_points = self.query.points.shape[1]
        if n_points == 0:
            raise ValueError("Query.points must not be empty")
        for track in self.tracks:
            if track.points.shape != (n_frames, n_points, 2):
                raise ValueError(f"Track.points must have shape {(n_frames, n_points, 2)}")
            if track.visibility.shape != (n_frames, n_points):
                raise ValueError(f"Track.visibility must have shape {(n_frames, n_points)}")
            if track.confidence.shape != (n_frames, n_points):
                raise ValueError(f"Track.confidence must have shape {(n_frames, n_points)}")
            if track.mask.shape != (n_frames, n_points):
                raise ValueError(f"Track.mask must have shape {(n_frames, n_points)}")
        for view_idx in range(n_views):
            frame_indices = self.query.frame_indices[view_idx]
            if frame_indices.min().item() < 0:
                raise ValueError("Query.frame_indices must be non-negative")
            if frame_indices.max().item() >= n_frames:
                raise ValueError("Query.frame_indices must be within the frames sequence")

    def save(self, folder: str) -> None:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
        self.query.save(os.path.join(folder, "query.pt"))
        for frame_idx, dataset in enumerate(self.frames):
            frame_dir = os.path.join(folder, f"frame{frame_idx}")
            cameras, images = {}, {}
            for view_idx in range(len(dataset)):
                camera = dataset[view_idx]
                name = f"{view_idx:04d}.png"
                os.makedirs(os.path.join(frame_dir, "images"), exist_ok=True)
                Image.fromarray(
                    (camera.ground_truth_image.detach().cpu()[:3].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                ).save(os.path.join(frame_dir, "images", name))
                if camera.ground_truth_image_mask is not None:
                    os.makedirs(os.path.join(frame_dir, "image_masks"), exist_ok=True)
                    tifffile.imwrite(os.path.join(frame_dir, "image_masks", name + ".tiff"),
                                     camera.ground_truth_image_mask.detach().cpu().float().numpy())
                if camera.ground_truth_depth is not None:
                    os.makedirs(os.path.join(frame_dir, "depths"), exist_ok=True)
                    tifffile.imwrite(os.path.join(frame_dir, "depths", name + ".tiff"),
                                     camera.ground_truth_depth.detach().cpu().float().numpy())
                if camera.ground_truth_depth_mask is not None:
                    os.makedirs(os.path.join(frame_dir, "depth_masks"), exist_ok=True)
                    tifffile.imwrite(os.path.join(frame_dir, "depth_masks", name + ".tiff"),
                                     camera.ground_truth_depth_mask.detach().cpu().float().numpy())
                camera_id = view_idx + 1
                cameras[camera_id] = ColmapCamera(
                    id=camera_id, model="PINHOLE", width=camera.image_width, height=camera.image_height,
                    params=np.array([
                        fov2focal(camera.FoVx, camera.image_width),
                        fov2focal(camera.FoVy, camera.image_height),
                        camera.image_width / 2, camera.image_height / 2,
                    ]),
                )
                images[camera_id] = ColmapImage(
                    id=camera_id, qvec=rotmat2qvec(camera.R.detach().cpu().numpy()),
                    tvec=camera.T.detach().cpu().numpy(), camera_id=camera_id, name=name,
                    xys=np.zeros((0, 2)), point3D_ids=np.zeros(0, dtype=np.int64),
                )
            os.makedirs(os.path.join(frame_dir, "sparse", "0"), exist_ok=True)
            write_model(cameras, images, {}, os.path.join(frame_dir, "sparse", "0"), ext=".txt")
        for view_idx, track in enumerate(self.tracks):
            track.save(os.path.join(folder, f"{view_idx:04d}.pt"))

    @classmethod
    def load(cls, folder: str, device="cpu") -> 'Supervision':
        query = Query.load(os.path.join(folder, "query.pt"), device=device)
        frames = []
        frame_idx = 0
        while os.path.isdir(os.path.join(folder, f"frame{frame_idx}")):
            frames.append(ColmapCameraDataset(os.path.join(folder, f"frame{frame_idx}")).to(device))
            frame_idx += 1
        tracks = [
            Track.load(os.path.join(folder, f"{view_idx:04d}.pt"), device=device)
            for view_idx in range(len(frames[0]))
        ]
        return cls(query=query, frames=frames, tracks=tracks)
