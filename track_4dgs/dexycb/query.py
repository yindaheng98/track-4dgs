"""Query and per-view tracks from multi-view 2D trajectories."""

from collections.abc import Sequence

import torch

from ..tracker.tracker import Query, Track


def query_and_tracks_from_trajectories(
        points: torch.Tensor,
        valid: torch.Tensor,
        widths: Sequence[int],
        heights: Sequence[int],
        start_indices: torch.Tensor,
):
    """Build a query and one track per view from ``[V, D, N, 2]`` pixels.

    ``start_indices`` is ``[V, N]`` and selects the query frame of each point
    in each view. ``valid`` marks annotated points. ``in_image`` requires the
    query pixel to be annotated and inside the view.
    """

    if points.ndim != 4 or points.shape[-1] != 2:
        raise ValueError("points must have shape [V, D, N, 2]")
    if valid.shape != points.shape[:3]:
        raise ValueError("valid must have shape [V, D, N]")
    if not torch.is_floating_point(points):
        raise TypeError("points must be floating point")
    if valid.dtype != torch.bool:
        raise TypeError("valid must be bool")
    n_views, n_frames, n_points, _ = points.shape
    if n_views == 0 or n_frames == 0 or n_points == 0:
        raise ValueError("points must contain at least one view, frame, and point")
    if len(widths) != n_views or len(heights) != n_views:
        raise ValueError("widths and heights must have one entry per view")
    if start_indices.shape != (n_views, n_points):
        raise ValueError("start_indices must have shape [V, N]")
    if start_indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("start_indices must be an integer tensor")
    if start_indices.min().item() < 0 or start_indices.max().item() >= n_frames:
        raise ValueError("start_indices must be within the trajectory frames")

    start_indices = start_indices.to(device=points.device)
    view_index = torch.arange(n_views, device=points.device)[:, None]
    point_index = torch.arange(n_points, device=points.device)
    seen = valid[view_index, start_indices, point_index]
    query_points = points[view_index, start_indices, point_index]
    in_image = torch.zeros_like(seen)
    for view_idx, (width, height) in enumerate(zip(widths, heights)):
        xy = query_points[view_idx]
        in_image[view_idx] = (
            seen[view_idx]
            & (xy[:, 0] >= 0)
            & (xy[:, 0] <= width - 1)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] <= height - 1)
        )
    query = Query(points=query_points, frame_indices=start_indices, in_image=in_image)
    confidence = valid.to(dtype=points.dtype)
    tracks = [
        Track(
            points=points[view_idx],
            visibility=confidence[view_idx],
            confidence=confidence[view_idx],
            mask=valid[view_idx],
        )
        for view_idx in range(n_views)
    ]
    return query, tracks
