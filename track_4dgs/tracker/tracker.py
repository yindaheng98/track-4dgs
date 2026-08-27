from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass(frozen=True)
class Query:
    """Point queries in pixel coordinates.

    ``points`` stores ``N`` pixel coordinates as ``[x, y]`` float pairs.
    ``frame_indices`` stores the source frame index for each point.
    """

    points: torch.Tensor
    frame_indices: torch.Tensor

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[-1] != 2:
            raise ValueError("Query.points must have shape [N, 2]")
        if not torch.is_floating_point(self.points):
            raise TypeError("Query.points must be a floating point tensor")
        if self.frame_indices.ndim != 1:
            raise ValueError("Query.frame_indices must have shape [N]")
        if self.frame_indices.shape[0] != self.points.shape[0]:
            raise ValueError("Query.frame_indices must have the same length as Query.points")
        if self.frame_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("Query.frame_indices must be an integer tensor")

    def to(self, device) -> 'Query':
        return Query(
            points=self.points.to(device),
            frame_indices=self.frame_indices.to(device),
        )


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


@dataclass(frozen=True)
class Track:
    """Tracked query locations and visibility over a frame sequence."""

    points: torch.Tensor
    visibility: torch.Tensor

    def __post_init__(self):
        if self.points.ndim != 3 or self.points.shape[-1] != 2:
            raise ValueError("Track.points must have shape [D, N, 2]")
        if not torch.is_floating_point(self.points):
            raise TypeError("Track.points must be a floating point tensor")
        if self.visibility.ndim != 2:
            raise ValueError("Track.visibility must have shape [D, N]")
        if self.visibility.shape != self.points.shape[:2]:
            raise ValueError("Track.visibility must match Track.points first two dimensions")

    def to(self, device) -> 'Track':
        return Track(
            points=self.points.to(device),
            visibility=self.visibility.to(device),
        )

    def __getitem__(self, index) -> Union[CameraTrack, 'Track']:
        points = self.points[index]
        visibility = self.visibility[index]
        if points.ndim == 2:
            return CameraTrack(points=points, visibility=visibility)
        if points.ndim == 3:
            return Track(points=points, visibility=visibility)
        raise TypeError("Track only supports indexing along the frame dimension")


class AbstractPointTracker(metaclass=ABCMeta):
    """Base class for point trackers over one camera sequence."""

    def to(self, device) -> 'AbstractPointTracker':
        return self

    def __call__(
            self,
            query: Query,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]],
            batch_size: Optional[int] = None) -> Track:
        """Validate inputs, run tracking, and validate the returned Track.

        ``frames`` and ``frame_masks`` are frame-major for one view:
        frames are ``[C, H, W]`` tensors, masks are optional ``[H, W]`` tensors.
        ``query`` contains ``N`` points on these frames, and the returned
        :class:`Track` must contain ``points`` with shape ``[D, N, 2]`` and
        ``visibility`` with shape ``[D, N]``, where ``D == len(frames)``.

        Query points are forwarded to :meth:`track` in chunks of ``batch_size``.
        ``None`` tracks all points in one call.
        """
        if len(frames) == 0:
            raise ValueError("frames must not be empty")
        if len(frame_masks) != len(frames):
            raise ValueError("frame_masks must have the same length as frames")

        for frame, frame_mask in zip(frames, frame_masks):
            if frame.ndim != 3:
                raise ValueError("frames entries must have shape [C, H, W]")
            if frame_mask is not None:
                if frame_mask.ndim != 2:
                    raise ValueError("frame_masks entries must have shape [H, W]")
                if frame_mask.shape != frame.shape[-2:]:
                    raise ValueError("frame_masks entries must match their frame spatial dimensions")

        if query.points.shape[0] == 0:
            raise ValueError("Query.points must not be empty")
        if query.frame_indices.min().item() < 0:
            raise ValueError("Query.frame_indices must be non-negative")
        if query.frame_indices.max().item() >= len(frames):
            raise ValueError("Query.frame_indices must be within the frames sequence")

        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        torch.cuda.empty_cache()
        n_points = query.points.shape[0]
        if batch_size is None or n_points <= batch_size:
            track = self.track(query, frames, frame_masks)
        else:
            tracks = []
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                batch_query = Query(
                    points=query.points[start:end],
                    frame_indices=query.frame_indices[start:end],
                )
                tracks.append(self.track(batch_query, frames, frame_masks))
                torch.cuda.empty_cache()
            track = Track(
                points=torch.cat([item.points for item in tracks], dim=1),
                visibility=torch.cat([item.visibility for item in tracks], dim=1),
            )
        torch.cuda.empty_cache()
        if track.points.shape != (len(frames), query.points.shape[0], 2):
            raise ValueError(f"AbstractPointTracker.track output points must have shape {(len(frames), query.points.shape[0], 2)}")
        if track.visibility.shape != (len(frames), query.points.shape[0]):
            raise ValueError(f"AbstractPointTracker.track output visibility must have shape {(len(frames), query.points.shape[0])}")
        return track

    @abstractmethod
    def track(
            self,
            query: Query,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]]) -> Track:
        """Track ``query`` points through ``frames`` for one view.

        Implementations should return a :class:`Track` whose ``points`` tensor
        has shape ``[len(frames), query.points.shape[0], 2]`` and whose
        ``visibility`` tensor has shape ``[len(frames), query.points.shape[0]]``.
        """
        raise NotImplementedError
