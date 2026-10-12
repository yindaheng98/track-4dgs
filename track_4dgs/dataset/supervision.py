from collections.abc import Sequence
from dataclasses import dataclass

from gaussian_splatting.dataset import CameraDataset

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
