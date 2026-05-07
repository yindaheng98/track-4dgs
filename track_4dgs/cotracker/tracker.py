from collections.abc import Sequence

import torch
from cotracker.predictor import CoTrackerPredictor

from track_4dgs.tracker import AbstractPointTracker, Query, Track


class Cotracker3PointTracker(AbstractPointTracker):
    """Track queried points with CoTracker3.

    Frames are expected to be ``[C, H, W]`` tensors and query points use pixel
    coordinates ``[x, y]`` in the original frame resolution.
    """

    def __init__(
            self,
            checkpoint: str = "./checkpoints/scaled_offline.pth",
            offline: bool = True):
        self.model = CoTrackerPredictor(checkpoint=checkpoint, offline=offline)
        self.model.eval()

    def to(self, device: torch.device) -> 'Cotracker3PointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    def track(self, query: Query, frames: Sequence[torch.Tensor], frame_masks: Sequence[torch.Tensor | None]) -> Track:
        assert all(frame.shape == frames[0].shape for frame in frames)

        video = torch.stack(frames, dim=0).unsqueeze(0)

        queries = torch.cat([query.frame_indices[:, None].to(dtype=query.points.dtype), query.points], dim=-1).unsqueeze(0)

        with torch.inference_mode():
            pred_tracks, pred_visibility = self.model(video, queries=queries)

        return Track(points=pred_tracks.squeeze(0), visibility=pred_visibility.squeeze(0))
