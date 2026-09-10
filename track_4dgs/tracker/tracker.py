from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import torch
from gaussian_splatting.dataset import CameraDataset


@dataclass(frozen=True)
class Query:
    """Point queries in pixel coordinates.

    ``points`` stores ``N`` corresponding pixel coordinates for each of ``V``
    views as ``[x, y]`` float pairs. ``frame_indices`` stores the source frame
    index for each point in each view. ``in_image`` is ``[V, N]`` bool
    validity (inside the image and, when known, in front of the camera).
    """

    points: torch.Tensor
    frame_indices: torch.Tensor
    in_image: torch.Tensor

    def __post_init__(self):
        if self.points.ndim != 3 or self.points.shape[-1] != 2:
            raise ValueError("Query.points must have shape [V, N, 2]")
        if not torch.is_floating_point(self.points):
            raise TypeError("Query.points must be a floating point tensor")
        if self.frame_indices.ndim != 2:
            raise ValueError("Query.frame_indices must have shape [V, N]")
        if self.frame_indices.shape != self.points.shape[:2]:
            raise ValueError("Query.frame_indices must match Query.points first two dimensions")
        if self.frame_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("Query.frame_indices must be an integer tensor")
        if self.in_image.ndim != 2:
            raise ValueError("Query.in_image must have shape [V, N]")
        if self.in_image.shape != self.points.shape[:2]:
            raise ValueError("Query.in_image must match Query.points first two dimensions")
        if self.in_image.dtype != torch.bool:
            raise TypeError("Query.in_image must be a boolean tensor")

    def to(self, device) -> 'Query':
        return Query(
            points=self.points.to(device),
            frame_indices=self.frame_indices.to(device),
            in_image=self.in_image.to(device),
        )


@dataclass(frozen=True)
class CameraTrack:
    """Track result for one camera in one frame."""

    points: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError("CameraTrack.points must have shape [N, 2]")
        if self.visibility.ndim != 1:
            raise ValueError("CameraTrack.visibility must have shape [N]")
        if self.visibility.shape[0] != self.points.shape[0]:
            raise ValueError("CameraTrack.visibility must match CameraTrack.points first dimension")
        if self.confidence.ndim != 1:
            raise ValueError("CameraTrack.confidence must have shape [N]")
        if self.confidence.shape[0] != self.points.shape[0]:
            raise ValueError("CameraTrack.confidence must match CameraTrack.points first dimension")
        if self.mask.ndim != 1:
            raise ValueError("CameraTrack.mask must have shape [N]")
        if self.mask.shape[0] != self.points.shape[0]:
            raise ValueError("CameraTrack.mask must match CameraTrack.points first dimension")
        if self.mask.dtype != torch.bool:
            raise TypeError("CameraTrack.mask must be a boolean tensor")

    def to(self, device) -> 'CameraTrack':
        return CameraTrack(
            points=self.points.to(device),
            visibility=self.visibility.to(device),
            confidence=self.confidence.to(device),
            mask=self.mask.to(device),
        )


@dataclass(frozen=True)
class Track:
    """Tracked query locations and validity over a frame sequence."""

    points: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self):
        if self.points.ndim != 3 or self.points.shape[-1] != 2:
            raise ValueError("Track.points must have shape [D, N, 2]")
        if not torch.is_floating_point(self.points):
            raise TypeError("Track.points must be a floating point tensor")
        if self.visibility.ndim != 2:
            raise ValueError("Track.visibility must have shape [D, N]")
        if self.visibility.shape != self.points.shape[:2]:
            raise ValueError("Track.visibility must match Track.points first two dimensions")
        if self.confidence.ndim != 2:
            raise ValueError("Track.confidence must have shape [D, N]")
        if self.confidence.shape != self.points.shape[:2]:
            raise ValueError("Track.confidence must match Track.points first two dimensions")
        if not torch.is_floating_point(self.confidence):
            raise TypeError("Track.confidence must be a floating point tensor")
        if self.mask.ndim != 2:
            raise ValueError("Track.mask must have shape [D, N]")
        if self.mask.shape != self.points.shape[:2]:
            raise ValueError("Track.mask must match Track.points first two dimensions")
        if self.mask.dtype != torch.bool:
            raise TypeError("Track.mask must be a boolean tensor")

    def to(self, device) -> 'Track':
        return Track(
            points=self.points.to(device),
            visibility=self.visibility.to(device),
            confidence=self.confidence.to(device),
            mask=self.mask.to(device),
        )

    def __getitem__(self, index) -> Union[CameraTrack, 'Track']:
        points = self.points[index]
        visibility = self.visibility[index]
        confidence = self.confidence[index]
        mask = self.mask[index]
        if points.ndim == 2:
            return CameraTrack(points=points, visibility=visibility, confidence=confidence, mask=mask)
        if points.ndim == 3:
            return Track(points=points, visibility=visibility, confidence=confidence, mask=mask)
        raise TypeError("Track only supports indexing along the frame dimension")


class AbstractPointTracker(metaclass=ABCMeta):
    """Base class for point trackers over multi-timestep camera datasets."""

    def to(self, device) -> 'AbstractPointTracker':
        return self

    def __call__(
            self,
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> Sequence[Track]:
        """Validate inputs, run tracking, and validate the returned tracks.

        ``query`` is view-major with one row per camera/view.
        ``frames`` is frame-major: one CameraDataset per frame, and
        each dataset is expected to contain cameras/views in the same order as
        ``query``. Returns one :class:`Track` per view.

        For each view, ``frames`` and ``frame_masks`` are frame-major:
        frames are ``[C, H, W]`` tensors, masks are optional ``[H, W]`` tensors.
        Each view in ``query`` contains ``N`` points on these frames, and the returned
        :class:`Track` must contain ``points`` with shape ``[D, N, 2]`` and
        ``visibility`` / ``confidence`` / ``mask`` with shape ``[D, N]``, where
        ``D == len(frames)``.

        Query points are forwarded to :meth:`track` in chunks of ``batch_size``.
        ``None`` tracks all points in one call.
        """
        if len(frames) == 0:
            return []
        n_views = len(frames[0])
        if any(len(dataset) != n_views for dataset in frames):
            raise ValueError("frames must all contain the same number of cameras")
        if query.points.shape[0] != n_views:
            raise ValueError("Query must have one row per camera/view")
        for view_idx in range(n_views):
            for dataset in frames:
                camera = dataset[view_idx]
                if camera.ground_truth_image is None:
                    raise ValueError("Point tracking requires cameras with loaded ground_truth_image tensors")
                frame = camera.ground_truth_image
                frame_mask = camera.ground_truth_image_mask
                if frame.ndim != 3:
                    raise ValueError("frames entries must have shape [C, H, W]")
                if frame_mask is not None:
                    if frame_mask.ndim != 2:
                        raise ValueError("frame_masks entries must have shape [H, W]")
                    if frame_mask.shape != frame.shape[-2:]:
                        raise ValueError("frame_masks entries must match their frame spatial dimensions")

            if query.points.shape[1] == 0:
                raise ValueError("Query.points must not be empty")
            if query.frame_indices[view_idx].min().item() < 0:
                raise ValueError("Query.frame_indices must be non-negative")
            if query.frame_indices[view_idx].max().item() >= len(frames):
                raise ValueError("Query.frame_indices must be within the frames sequence")

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        view_tracks = self.track(query, frames, batch_size)
        if len(view_tracks) != n_views:
            raise ValueError(f"AbstractPointTracker.track must return one Track per view, got {len(view_tracks)}")
        n_points = query.points.shape[1]
        for track in view_tracks:
            if track.points.shape != (len(frames), n_points, 2):
                raise ValueError(f"Track.points must have shape {(len(frames), n_points, 2)}")
            if track.visibility.shape != (len(frames), n_points):
                raise ValueError(f"Track.visibility must have shape {(len(frames), n_points)}")
            if track.confidence.shape != (len(frames), n_points):
                raise ValueError(f"Track.confidence must have shape {(len(frames), n_points)}")
            if track.mask.shape != (len(frames), n_points):
                raise ValueError(f"Track.mask must have shape {(len(frames), n_points)}")
        return view_tracks

    @abstractmethod
    def track(
            self,
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> Sequence[Track]:
        """Track ``query`` across ``frames``.

        Implementations should return one :class:`Track` per view.
        """
        raise NotImplementedError
