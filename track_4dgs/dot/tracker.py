from collections.abc import Sequence

import torch
from dot.models.dense_optical_tracking import DenseOpticalTracker
from dot.utils.torch import get_grid

from track_4dgs.tracker import AbstractViewPointTracker, Track

# https://github.com/yindaheng98/TrackerSplat/blob/main/trackersplat/motionestimator/point_tracker/dot.py
# ``dot`` is the original DOT initializer (CoTracker2). The other names swap that initializer.
MODELS = {
    "dot": (
        "submodules/dot/configs/cotracker2_patch_4_wind_8.json",
        "checkpoints/movi_f_cotracker2_patch_4_wind_8.pth",
    ),
    "cotracker3": (
        "submodules/dot/configs/cotracker3_wind_60.json",
        "checkpoints/scaled_offline.pth",
    ),
    "tapir": (
        "submodules/dot/configs/tapir.json",
        "checkpoints/panning_movi_e_tapir.pth",
    ),
    "bootstapir": (
        "submodules/dot/configs/bootstapir.json",
        "checkpoints/panning_movi_e_plus_bootstapir.pth",
    ),
}
ESTIMATOR_CONFIG = "submodules/dot/configs/raft_patch_8.json"
ESTIMATOR_CHECKPOINT = "checkpoints/cvo_raft_patch_8.pth"
REFINER_CONFIG = "submodules/dot/configs/raft_patch_4_alpha.json"
REFINER_CHECKPOINT = "checkpoints/movi_f_raft_patch_4_alpha.pth"
# TrackerSplat runs 1024 sparse tracks in one batch.
NUM_TRACKS = 1024
SIM_TRACKS = 1024


def resize_model(model: DenseOpticalTracker, height: int, width: int, device: torch.device) -> DenseOpticalTracker:
    """Point the fixed estimator and refiner grids at this frame size.

    Both sides are rounded down to a multiple of 8, matching TrackerSplat's
    ``rescale_factor=1`` path. The estimator patch is 8 and the refiner patch is 4.
    """
    height = height // 8 * 8
    width = width // 8 * 8
    if height < 8 or width < 8:
        raise ValueError("DOTPointTracker needs both sides at least 8")
    model.resolution = [height, width]
    model.point_tracker.optical_flow_estimator.register_buffer(
        "coarse_grid", get_grid(height // 8, width // 8, device=device),
    )
    model.optical_flow_refiner.register_buffer(
        "coarse_grid", get_grid(height // 4, width // 4, device=device),
    )
    return model


def load_dot(model: str = "dot"):
    """Load one DOT variant.

    ``model`` is ``dot``, ``cotracker3``, ``tapir``, or ``bootstapir``.
    Estimator and refiner checkpoints stay the same for every variant.
    """
    if model not in MODELS:
        raise ValueError("Please choose from [dot, cotracker3, tapir, bootstapir]")
    tracker_config, tracker_path = MODELS[model]
    loaded = DenseOpticalTracker(
        tracker_config=tracker_config,
        tracker_path=tracker_path,
        estimator_config=ESTIMATOR_CONFIG,
        estimator_path=ESTIMATOR_CHECKPOINT,
        refiner_config=REFINER_CONFIG,
        refiner_path=REFINER_CHECKPOINT,
    )
    loaded.eval()
    return loaded


class DOTPointTracker(AbstractViewPointTracker):
    """Track queried points with DOT.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]`` and share one
    resolution. Query points are ``[x, y]`` pixels in their source frames.
    ``model`` selects the sparse initializer: ``dot`` (CoTracker2),
    ``cotracker3``, ``tapir``, or ``bootstapir``.

    One call runs ``tracks_for_queries``. DOT tracks 1024 sparse points,
    refines a dense flow from each source frame, and samples that flow at the
    queries. The working resolution follows the frame, rounded down to a
    multiple of 8. Queries and predictions stay in the original pixel frame.

    DOT queries are ``[t, y, x]``. Returned tracks are ``[x, y, visibility]``.
    ``visibility`` and ``confidence`` are that score. The refiner thresholds
    it, so both are 0 or 1, and 1 on the source frame.
    """

    def __init__(self, model: str = "dot"):
        self.model = load_dot(model)
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> "DOTPointTracker":
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def track_view(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[torch.Tensor | None]) -> Track:
        del frame_masks
        if any(frame.shape[0] != 3 for frame in frames):
            raise ValueError("DOTPointTracker expects RGB frames with shape [3, H, W]")
        if any(frame.shape[-2:] != frames[0].shape[-2:] for frame in frames):
            raise ValueError("DOTPointTracker expects all frames in a view to have the same resolution")
        if points.shape[0] == 0:
            raise ValueError("DOTPointTracker expects at least one query point")
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= len(frames):
            raise ValueError("frame_indices must point at frames in the sequence")

        _, height, width = frames[0].shape
        self.model = resize_model(self.model, height, width, self.device)
        video = torch.stack([
            frame.detach().clamp(0, 1).to(device=self.device, dtype=torch.float32)
            for frame in frames
        ], dim=0).unsqueeze(0)
        query_xy = points.to(device=self.device, dtype=torch.float32)
        query_t = frame_indices.to(device=self.device, dtype=torch.float32)
        # submodules/dot/dot/models/dense_optical_tracking.py#L100-L104
        # query_points is [B, S, 3] as [t, y, x] in the original frame's pixels.
        query_points = torch.stack([query_t, query_xy[:, 1], query_xy[:, 0]], dim=-1).unsqueeze(0)
        prediction = self.model(
            {"video": video, "query_points": query_points},
            mode="tracks_for_queries",
            num_tracks=NUM_TRACKS,
            sim_tracks=SIM_TRACKS,
        )
        # dense_to_sparse_tracks writes [x, y, visibility] in original pixels.
        # submodules/dot/dot/models/dense_optical_tracking.py#L213-L216
        tracks = prediction["tracks"][0]
        pred = tracks[..., :2].to(device=points.device, dtype=points.dtype)
        visibility = tracks[..., 2].to(device=points.device, dtype=points.dtype).clamp(0, 1)
        return Track(
            points=pred,
            visibility=visibility,
            confidence=visibility.clone(),
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=pred.device),
        )
