from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset
from tqdm import tqdm

from .tracker import AbstractPointTracker, Query, Track


class AbstractViewPointTracker(AbstractPointTracker):
    """Point tracker that processes one camera sequence at a time."""

    tqdm_position = 0
    tqdm_leave = True

    def track(
            self,
            query: Query,
            frames: Sequence[CameraDataset]) -> Sequence[Track]:
        view_tracks = []
        for view_idx in tqdm(range(query.points.shape[0]), desc="Tracking views", position=self.tqdm_position, leave=self.tqdm_leave, dynamic_ncols=True):
            views = []
            frame_masks = []
            for dataset in frames:
                camera = dataset[view_idx]
                views.append(camera.ground_truth_image)
                frame_masks.append(camera.ground_truth_image_mask)

            points = query.points[view_idx]
            frame_indices = query.frame_indices[view_idx]
            valid = query.in_image[view_idx]

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
                )
                result_points[:, valid] = track.points
                result_visibility[:, valid] = track.visibility
                result_confidence[:, valid] = track.confidence
                result_mask[:, valid] = track.mask

            view_tracks.append(Track(
                points=result_points,
                visibility=result_visibility,
                confidence=result_confidence,
                mask=result_mask,
            ))
        return view_tracks

    @abstractmethod
    def track_view(
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
