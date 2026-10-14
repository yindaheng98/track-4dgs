from collections.abc import Sequence

import torch
from cotracker.predictor import CoTrackerPredictor

from track_4dgs.tracker import AbstractViewPointTracker, Track


class CoTrackerPredictorWithConfidence(CoTrackerPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_forward = self.model.forward

        def forward(*args, **kwargs):
            tracks, vis, self.confidence, train_data = model_forward(*args, **kwargs)
            return tracks, vis, self.confidence, train_data

        self.model.forward = forward

    def forward(self, *args, **kwargs):
        tracks, visibilities = super().forward(*args, **kwargs)
        return tracks, visibilities, self.confidence[:, :, : tracks.shape[2]]


class Cotracker3PointTracker(AbstractViewPointTracker):
    """Track queried points with CoTracker3.

    Frames are expected to be ``[C, H, W]`` tensors and query points use pixel
    coordinates ``[x, y]`` in the original frame resolution.

    ``visibility`` is the predictor's thresholded occlusion flag.
    ``confidence`` is the inner-model sigmoid score in ``(0, 1)``, trained as
    ``P(error <= 12 px)``.

    ``batch_size`` chunks query points after the video encoder. ``None`` tracks
    every point in one call.
    """

    def __init__(
            self,
            checkpoint: str = "./checkpoints/scaled_offline.pth",
            offline: bool = True,
            batch_size: int | None = None):
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.model = CoTrackerPredictorWithConfidence(checkpoint=checkpoint, offline=offline)
        self.model.eval()
        self.batch_size = batch_size

    def to(self, device: torch.device) -> 'Cotracker3PointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    def track_batch(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[torch.Tensor | None]) -> Track:
        assert all(frame.shape == frames[0].shape for frame in frames)

        video = torch.stack(frames, dim=0).unsqueeze(0)
        queries = torch.cat([frame_indices[:, None].to(dtype=points.dtype), points], dim=-1).unsqueeze(0)

        with torch.inference_mode():
            n_points = queries.shape[1]
            if self.batch_size is None or n_points <= self.batch_size:
                pred_tracks, pred_visibility, pred_confidence = self.model(video, queries=queries)
            else:
                # fnet reads only the video, but CoTracker3 calls it from inside forward.
                # The first point chunk runs fnet and stores every output. Each later
                # chunk hands those tensors back in the same order.
                fnet = self.model.model.fnet
                fnet_forward = fnet.forward
                encoded = []

                def fnet_record(*args, **kwargs):
                    features = fnet_forward(*args, **kwargs)
                    encoded.append(features)
                    return features

                replay_index = 0

                def fnet_replay(*args, **kwargs):
                    nonlocal replay_index
                    features = encoded[replay_index]
                    replay_index += 1
                    return features

                tracks, visibilities, confidences = [], [], []
                try:
                    fnet.forward = fnet_record
                    for start in range(0, n_points, self.batch_size):
                        end = min(start + self.batch_size, n_points)
                        pred_tracks, pred_visibility, pred_confidence = self.model(video, queries=queries[:, start:end])
                        tracks.append(pred_tracks)
                        visibilities.append(pred_visibility)
                        confidences.append(pred_confidence)
                        replay_index = 0
                        fnet.forward = fnet_replay
                finally:
                    fnet.forward = fnet_forward
                pred_tracks = torch.cat(tracks, dim=2)
                pred_visibility = torch.cat(visibilities, dim=2)
                pred_confidence = torch.cat(confidences, dim=2)

        pred_tracks = pred_tracks.squeeze(0)
        pred_visibility = pred_visibility.squeeze(0)
        pred_confidence = pred_confidence.squeeze(0)
        return Track(
            points=pred_tracks,
            visibility=pred_visibility.to(dtype=points.dtype),
            confidence=pred_confidence.to(dtype=points.dtype),
            mask=torch.ones(pred_visibility.shape, dtype=torch.bool, device=pred_tracks.device),
        )
