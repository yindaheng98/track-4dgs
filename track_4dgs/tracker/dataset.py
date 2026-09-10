from collections.abc import Sequence
from typing import Optional

from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.camera import Camera

from .tracker import AbstractPointTracker, CameraTrack, Query


class TrackedCameraDataset(CameraDataset):
    """CameraDataset wrapper that attaches per-camera track results."""

    def __init__(self, dataset: CameraDataset, camera_tracks: Sequence[CameraTrack]):
        if len(camera_tracks) != len(dataset):
            raise ValueError("camera_tracks must have the same length as dataset")
        self.dataset = dataset
        self.camera_tracks = list(camera_tracks)

    @property
    def __class__(self):
        return self.dataset.__class__

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def to(self, device) -> 'TrackedCameraDataset':
        self.dataset = self.dataset.to(device)
        self.camera_tracks = [track.to(device) for track in self.camera_tracks]
        return self

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Camera:
        camera = self.dataset[idx]
        return camera._replace(custom_data={
            **camera.custom_data,
            "track": self.camera_tracks[idx],
        })


class CameraDatasetTracker:
    """Run a point tracker and attach tracks to camera datasets."""

    def __init__(self, tracker: AbstractPointTracker):
        self.tracker = tracker

    def to(self, device) -> 'CameraDatasetTracker':
        self.tracker = self.tracker.to(device)
        return self

    def __call__(
            self,
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> list[TrackedCameraDataset]:
        """Track points for a frame-major collection of camera datasets.

        ``query`` is view-major: one row per camera/view.
        ``frames`` is frame-major: one CameraDataset per frame, and
        each dataset is expected to contain cameras/views in the same order as
        ``query``.  The return value keeps the frame-major layout.
        ``batch_size`` is forwarded to the underlying point tracker.
        """
        frames = list(frames)
        if len(frames) == 0:
            return []

        view_tracks = self.tracker(query, frames, batch_size=batch_size)
        frame_camera_tracks = [[] for _ in frames]
        for track in view_tracks:
            for frame_idx, camera_tracks in enumerate(frame_camera_tracks):
                camera_tracks.append(track[frame_idx])

        return [
            TrackedCameraDataset(
                dataset=dataset,
                camera_tracks=camera_tracks,
            )
            for dataset, camera_tracks in zip(frames, frame_camera_tracks)
        ]
