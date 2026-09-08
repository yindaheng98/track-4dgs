from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset
from tqdm import tqdm

from .tracker import AbstractPointTracker, Query, Track


class AbstractViewPointTracker(AbstractPointTracker):
    """Point tracker that processes one camera sequence at a time."""

    def track(
            self,
            view_queries: Sequence[Query],
            frame_datasets: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> Sequence[Track]:
        view_tracks = []
        for view_idx, query in enumerate(tqdm(view_queries, desc="Tracking views")):
            frames = []
            frame_masks = []
            for dataset in frame_datasets:
                camera = dataset[view_idx]
                frames.append(camera.ground_truth_image)
                frame_masks.append(camera.ground_truth_image_mask)
            view_tracks.append(self.track_view(query, frames, frame_masks, batch_size=batch_size))
        return view_tracks

    def track_view(
            self,
            query: Query,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]],
            batch_size: Optional[int] = None) -> Track:
        """Track ``query`` points through ``frames`` for one view.

        Query points are forwarded to :meth:`track_batch` in chunks of ``batch_size``.
        ``None`` tracks all points in one call.
        """
        torch.cuda.empty_cache()
        n_points = query.points.shape[0]
        if batch_size is None or n_points <= batch_size:
            track = self.track_batch(query, frames, frame_masks)
        else:
            tracks: list[Track] = []
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                batch_query = Query(
                    points=query.points[start:end],
                    frame_indices=query.frame_indices[start:end],
                )
                tracks.append(self.track_batch(batch_query, frames, frame_masks))
                torch.cuda.empty_cache()
            track = Track(
                points=torch.cat([item.points for item in tracks], dim=1),
                visibility=torch.cat([item.visibility for item in tracks], dim=1),
            )
        torch.cuda.empty_cache()
        return track

    @abstractmethod
    def track_batch(
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
