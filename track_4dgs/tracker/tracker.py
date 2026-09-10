from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import torch
from gaussian_splatting.dataset import CameraDataset

from ..utils import project_points


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

    @classmethod
    def from_projection(
            cls,
            xyz: torch.Tensor,
            frames: Sequence[CameraDataset],
            frame_indices: torch.Tensor) -> 'Query':
        """Build a query by projecting world points into ``frames``.

        ``xyz`` is ``[N, 3]``. ``frames`` is frame-major. ``frame_indices``
        is ``[N]`` and names the video frame each point comes from; every view
        uses that frame's camera. ``in_image`` comes from
        :func:`~track_4dgs.utils.project_points`.
        """
        if xyz.ndim != 2 or xyz.shape[-1] != 3:
            raise ValueError("xyz must have shape [N, 3]")
        if frame_indices.ndim != 1:
            raise ValueError("frame_indices must have shape [N]")
        if frame_indices.shape[0] != xyz.shape[0]:
            raise ValueError("frame_indices must match xyz point count N")
        n_views = len(frames[0])
        if any(len(dataset) != n_views for dataset in frames):
            raise ValueError("frames must all contain the same number of cameras")
        points = []
        in_image = []
        for view_idx in range(n_views):
            uv = xyz.new_empty((xyz.shape[0], 2))
            valid = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
            for frame_idx in frame_indices.unique().tolist():
                sel = frame_indices == frame_idx
                pixels, in_frustum = project_points(xyz[sel], frames[frame_idx][view_idx])
                uv[sel] = pixels
                valid[sel] = in_frustum
            points.append(uv)
            in_image.append(valid)
        return cls(
            points=torch.stack(points),
            frame_indices=frame_indices.unsqueeze(0).expand(n_views, -1).contiguous(),
            in_image=torch.stack(in_image),
        )


@dataclass(frozen=True)
class CameraTrack:
    """Track result for one camera in one frame.

    Field semantics match :class:`Track` at a single frame: ``points`` are
    ``[N, 2]`` pixels, ``visibility`` / ``confidence`` are ``[N]``, and
    ``mask`` is ``[N]`` bool validity.
    """

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
    """Tracked query locations and validity over a frame sequence.

    ``points`` is ``[D, N, 2]`` pixel coordinates. ``visibility``,
    ``confidence``, and ``mask`` are ``[D, N]``.

    ``visibility`` is the occlusion score: whether the point is visible in that
    frame. Some models keep a float in ``(0, 1)``; CoTracker3's predictor
    thresholds it to a bool.

    ``confidence`` is a localization score in ``(0, 1)``, not a calibrated
    variance. Models emit unbounded logits (iterative residuals for CoTracker3
    / MV-TAP, a linear head for VGGT) then apply ``sigmoid``. Training is
    binary: 1 if the predicted location is within a pixel threshold of ground
    truth, else 0. CoTracker3 and MV-TAP use 12 px; VGGT uses 3 px. The
    stored value is therefore ``P(error < threshold)``. Well-tracked points
    often saturate near 1.

    ``mask`` is implementation-defined boolean validity for each point.
    """

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
