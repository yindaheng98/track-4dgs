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
            query: Query,
            frames: Sequence[CameraDataset],
            batch_size: Optional[int] = None) -> Sequence[Track]:
        view_tracks = []
        for view_idx in tqdm(range(query.points.shape[0]), desc="Tracking views"):
            views = []
            frame_masks = []
            for dataset in frames:
                camera = dataset[view_idx]
                views.append(camera.ground_truth_image)
                frame_masks.append(camera.ground_truth_image_mask)

            points = query.points[view_idx]
            frame_indices = query.frame_indices[view_idx]
            sizes = points.new_tensor([
                [views[frame_idx].shape[-1], views[frame_idx].shape[-2]]
                for frame_idx in frame_indices.tolist()
            ])
            valid = (
                (points[:, 0] >= 0) & (points[:, 0] < sizes[:, 0])
                & (points[:, 1] >= 0) & (points[:, 1] < sizes[:, 1])
            )

            n_frames = len(views)
            result_points = points.unsqueeze(0).expand(n_frames, -1, -1).clone()
            result_visibility = points.new_zeros((n_frames, points.shape[0]))
            result_confidence = points.new_zeros((n_frames, points.shape[0]))
            result_mask = torch.zeros((n_frames, points.shape[0]), dtype=torch.bool, device=points.device)

            if valid.any():
                track = self.track_view(
                    points[valid],
                    frame_indices[valid],
                    views,
                    frame_masks,
                    batch_size=batch_size,
                )
                in_view = torch.stack([
                    (frame_points[:, 0] >= 0)
                    & (frame_points[:, 0] < frame.shape[-1])
                    & (frame_points[:, 1] >= 0)
                    & (frame_points[:, 1] < frame.shape[-2])
                    for frame_points, frame in zip(track.points, views)
                ])
                result_points[:, valid] = track.points
                result_visibility[:, valid] = track.visibility
                result_confidence[:, valid] = track.confidence
                result_mask[:, valid] = track.mask & in_view

            view_tracks.append(Track(
                points=result_points,
                visibility=result_visibility,
                confidence=result_confidence,
                mask=result_mask,
            ))
        return view_tracks

    def track_view(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]],
            batch_size: Optional[int] = None) -> Track:
        """Track query points through ``frames`` for one view.

        Points are forwarded to :meth:`track_batch` in chunks of ``batch_size``.
        ``None`` tracks all points in one call.
        """
        torch.cuda.empty_cache()
        if batch_size is None or points.shape[0] <= batch_size:
            track = self.track_batch(points, frame_indices, frames, frame_masks)
        else:
            tracks: list[Track] = []
            for start in range(0, points.shape[0], batch_size):
                end = min(start + batch_size, points.shape[0])
                tracks.append(self.track_batch(
                    points[start:end],
                    frame_indices[start:end],
                    frames,
                    frame_masks,
                ))
                torch.cuda.empty_cache()
            track = Track(
                points=torch.cat([item.points for item in tracks], dim=1),
                visibility=torch.cat([item.visibility for item in tracks], dim=1),
                confidence=torch.cat([item.confidence for item in tracks], dim=1),
                mask=torch.cat([item.mask for item in tracks], dim=1),
            )
        torch.cuda.empty_cache()
        return track

    @abstractmethod
    def track_batch(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[Optional[torch.Tensor]]) -> Track:
        """Track query points through ``frames`` for one view.

        Implementations should return a :class:`Track` whose ``points`` tensor
        has shape ``[len(frames), points.shape[0], 2]`` and whose
        ``visibility`` / ``confidence`` / ``mask`` tensors have shape
        ``[len(frames), points.shape[0]]``.
        """
        raise NotImplementedError
