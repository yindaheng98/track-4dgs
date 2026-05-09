from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.camera import Camera

from .tracker import AbstractPointTracker, Query


@dataclass(frozen=True)
class CameraTrack:
    """Track result for one camera in one frame."""

    points: torch.Tensor
    visibility: torch.Tensor

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError("CameraTrack.points must have shape [N, 2]")
        if self.visibility.ndim != 1:
            raise ValueError("CameraTrack.visibility must have shape [N]")
        if self.visibility.shape[0] != self.points.shape[0]:
            raise ValueError("CameraTrack.visibility must match CameraTrack.points first dimension")

    def to(self, device) -> 'CameraTrack':
        return CameraTrack(
            points=self.points.to(device),
            visibility=self.visibility.to(device),
        )


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
    """Run a point tracker for each view and attach tracks to camera datasets."""

    def __init__(self, tracker: AbstractPointTracker):
        self.tracker = tracker

    def to(self, device) -> 'CameraDatasetTracker':
        self.tracker = self.tracker.to(device)
        return self

    def __call__(
            self,
            view_queries: Iterable[Query],
            frame_datasets: Iterable[CameraDataset]) -> list[TrackedCameraDataset]:
        """Track points for a frame-major collection of camera datasets.

        ``view_queries`` is view-major: one query per camera/view.
        ``frame_datasets`` is frame-major: one CameraDataset per frame, and
        each dataset is expected to contain cameras/views in the same order as
        ``view_queries``.  The return value keeps the frame-major layout.
        """
        view_queries = list(view_queries)
        frame_datasets = list(frame_datasets)
        if len(frame_datasets) == 0:
            return []

        frame_camera_tracks = [[] for _ in frame_datasets]
        for view_idx, query in enumerate(view_queries):
            frames = []
            frame_masks = []
            for dataset in frame_datasets:
                camera = dataset[view_idx]
                if camera.ground_truth_image is None:
                    raise ValueError("Point tracking requires cameras with loaded ground_truth_image tensors")
                frames.append(camera.ground_truth_image)
                frame_masks.append(camera.ground_truth_image_mask)
            track = self.tracker(query, frames, frame_masks)
            for frame_idx, camera_tracks in enumerate(frame_camera_tracks):
                camera_tracks.append(CameraTrack(
                    points=track.points[frame_idx],
                    visibility=track.visibility[frame_idx],
                ))

        return [
            TrackedCameraDataset(
                dataset=dataset,
                camera_tracks=camera_tracks,
            )
            for dataset, camera_tracks in zip(frame_datasets, frame_camera_tracks)
        ]
