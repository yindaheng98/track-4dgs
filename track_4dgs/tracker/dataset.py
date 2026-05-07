from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional

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
