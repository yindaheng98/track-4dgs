from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

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


class AbstractPointTracker(metaclass=ABCMeta):
    """Base class for point trackers over one camera sequence."""

    def to(self, device) -> 'AbstractPointTracker':
        return self

    def __call__(
            self,
            query: Query,
            ground_truth_images: Sequence[torch.Tensor],
            ground_truth_image_masks: Sequence[Optional[torch.Tensor]]) -> Track:
        if len(ground_truth_images) == 0:
            raise ValueError("ground_truth_images must not be empty")
        if len(ground_truth_image_masks) != len(ground_truth_images):
            raise ValueError("ground_truth_image_masks must have the same length as ground_truth_images")

        for image, mask in zip(ground_truth_images, ground_truth_image_masks):
            if image.ndim != 3:
                raise ValueError("ground_truth_images entries must have shape [C, H, W]")
            if mask is not None:
                if mask.ndim != 2:
                    raise ValueError("ground_truth_image_masks entries must have shape [H, W]")
                if mask.shape != image.shape[-2:]:
                    raise ValueError("ground_truth_image_masks entries must match their image spatial dimensions")

        if query.frame_indices.numel() > 0:
            if query.frame_indices.min().item() < 0:
                raise ValueError("Query.frame_indices must be non-negative")
            if query.frame_indices.max().item() >= len(ground_truth_images):
                raise ValueError("Query.frame_indices must be within the ground_truth_images sequence")

        track = self.track(query, ground_truth_images, ground_truth_image_masks)
        if track.points.shape != (len(ground_truth_images), query.points.shape[0], 2):
            raise ValueError(f"AbstractPointTracker.track output points must have shape {(len(ground_truth_images), query.points.shape[0], 2)}")
        if track.visibility.shape != (len(ground_truth_images), query.points.shape[0]):
            raise ValueError(f"AbstractPointTracker.track output visibility must have shape {(len(ground_truth_images), query.points.shape[0])}")
        return track

    @abstractmethod
    def track(
            self,
            query: Query,
            ground_truth_images: Sequence[torch.Tensor],
            ground_truth_image_masks: Sequence[Optional[torch.Tensor]]) -> Track:
        """Return tracked pixel coordinates and visibility."""
        raise NotImplementedError
