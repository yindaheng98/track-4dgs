import os
from collections.abc import Sequence

import torch
from vggt.dependency.vggsfm_tracker import TrackerPredictor

from track_4dgs.tracker import AbstractViewPointTracker, Track
from track_4dgs.vggt.tracker import padding_square, points_from_square, points_to_square


def load_vggsfm(checkpoint: str = "checkpoints/vggsfm_v2_tracker.pt") -> TrackerPredictor:
    model = TrackerPredictor()
    if os.path.isfile(checkpoint):
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        model.load_state_dict(torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt",
            map_location="cpu",
        ))
    return model


class VGGSfMPointTracker(AbstractViewPointTracker):
    """Track queried points with the VGGSfM ``TrackerPredictor``.

    Frames are expected to be RGB ``[3, H, W]`` tensors in ``[0, 1]`` and query
    points use pixel coordinates ``[x, y]`` in the source frame. Images are
    center-padded to a square and resized with the same helpers as
    :class:`~track_4dgs.vggt.tracker.VGGTPointTracker`. The tracker reads query
    features from frame 0, so each distinct query frame is swapped to the front
    before prediction and swapped back afterwards.

    ``visibility`` is the coarse predictor's sigmoid score in ``(0, 1)``.
    ``confidence`` is 1: fine refinement is run with ``compute_score=False``,
    so the released tracker does not emit a localization score.
    """

    def __init__(
            self,
            checkpoint: str = "checkpoints/vggsfm_v2_tracker.pt",
            img_load_resolution: int = 1024,
            coarse_iters: int = 6,
            fine_tracking: bool = True,
            max_points_num: int = 163840,
            fine_chunk: int = 40960):
        if img_load_resolution % 8 != 0:
            raise ValueError("img_load_resolution must be divisible by 8")
        if max_points_num <= 0:
            raise ValueError("max_points_num must be a positive integer")
        self.model = load_vggsfm(checkpoint)
        self.model.eval()
        self.img_load_resolution = img_load_resolution
        self.coarse_iters = coarse_iters
        self.fine_tracking = fine_tracking
        self.max_points_num = max_points_num
        self.fine_chunk = fine_chunk
        self.device = torch.device("cpu")

    def to(self, device) -> 'VGGSfMPointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self
