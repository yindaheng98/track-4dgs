from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from .tracker import AbstractPointTracker, Query, Track


class AbstractMultiViewPointTracker(AbstractPointTracker):
    """Point tracker that consumes all camera views jointly."""

    def track(
            self,
            view_queries: Sequence[Query],
            frame_datasets: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> Sequence[Track]:
        n_points = view_queries[0].points.shape[0]
        torch.cuda.empty_cache()
        if batch_size is None or n_points <= batch_size:
            tracks = self.track_batch(view_queries, frame_datasets)
        else:
            tracks: list[Sequence[Track]] = []
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                batch_queries = [
                    Query(
                        points=query.points[start:end],
                        frame_indices=query.frame_indices[start:end],
                    )
                    for query in view_queries
                ]
                tracks.append(self.track_batch(batch_queries, frame_datasets))
                torch.cuda.empty_cache()
            tracks = [
                Track(
                    points=torch.cat([item[view_idx].points for item in tracks], dim=1),
                    visibility=torch.cat([item[view_idx].visibility for item in tracks], dim=1),
                    confidence=torch.cat([item[view_idx].confidence for item in tracks], dim=1),
                )
                for view_idx in range(len(view_queries))
            ]
        torch.cuda.empty_cache()
        return tracks

    @abstractmethod
    def track_batch(
            self,
            view_queries: Sequence[Query],
            frame_datasets: Sequence[CameraDataset]) -> Sequence[Track]:
        """Track ``view_queries`` jointly across ``frame_datasets``.

        Implementations should return one :class:`Track` per view, each with
        ``points`` shape ``[len(frame_datasets), N, 2]`` and ``visibility`` /
        ``confidence`` shape ``[len(frame_datasets), N]``.
        """
        raise NotImplementedError
