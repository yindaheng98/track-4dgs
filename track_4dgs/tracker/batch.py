from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from .tracker import AbstractPointTracker, Query, Track


class AbstractBatchPointTracker(AbstractPointTracker):
    """Point tracker that consumes all camera views jointly, in point batches."""

    def track(
            self,
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> Sequence[Track]:
        n_points = query.points.shape[1]
        torch.cuda.empty_cache()
        if batch_size is None or n_points <= batch_size:
            tracks = self.track_batch(query, frames)
        else:
            tracks: list[Sequence[Track]] = []
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                tracks.append(self.track_batch(
                    Query(
                        points=query.points[:, start:end],
                        frame_indices=query.frame_indices[:, start:end],
                        in_image=query.in_image[:, start:end],
                    ),
                    frames,
                ))
                torch.cuda.empty_cache()
            tracks: Sequence[Track] = [
                Track(
                    points=torch.cat([item[view_idx].points for item in tracks], dim=1),
                    visibility=torch.cat([item[view_idx].visibility for item in tracks], dim=1),
                    confidence=torch.cat([item[view_idx].confidence for item in tracks], dim=1),
                    mask=torch.cat([item[view_idx].mask for item in tracks], dim=1),
                )
                for view_idx in range(query.points.shape[0])
            ]
        torch.cuda.empty_cache()
        return [
            Track(
                points=track.points,
                visibility=track.visibility,
                confidence=track.confidence,
                mask=track.mask & query.in_image[view_idx].unsqueeze(0),
            )
            for view_idx, track in enumerate(tracks)
        ]

    @abstractmethod
    def track_batch(
            self,
            query: Query,
            frames: Sequence[CameraDataset]) -> Sequence[Track]:
        """Track ``query`` jointly across ``frames``.

        Implementations should return one :class:`Track` per view, each with
        ``points`` shape ``[len(frames), N, 2]`` and ``visibility`` /
        ``confidence`` / ``mask`` shape ``[len(frames), N]``.
        """
        raise NotImplementedError
