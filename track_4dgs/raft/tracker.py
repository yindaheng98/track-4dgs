from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

from track_4dgs.tracker import AbstractViewPointTracker, Track

# https://docs.pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html
# raft_large(pretrained=True) is Raft_Large_Weights.C_T_SKHT_V2.
MODEL_BUILDERS = {
    "large": (raft_large, Raft_Large_Weights.DEFAULT),
    "small": (raft_small, Raft_Small_Weights.DEFAULT),
}


def load_raft(model: str = "large"):
    """Load a pretrained torchvision RAFT model.

    ``model`` is ``large`` (``raft_large``) or ``small`` (``raft_small``).
    Weights download on first use. The model returns a list of flow iterates;
    the last one is the prediction used below.
    """
    if model not in MODEL_BUILDERS:
        raise ValueError("Please choose from [large, small]")
    builder, weights = MODEL_BUILDERS[model]
    loaded = builder(weights=weights, progress=False)
    loaded.eval()
    return loaded


def preprocess(image: torch.Tensor) -> torch.Tensor:
    """Map one RGB ``[3, H, W]`` frame in ``[0, 1]`` to the RAFT input.

    The tutorial maps ``[0, 1]`` to ``[-1, 1]`` with mean ``0.5`` and std
    ``0.5``. H and W must be divisible by 8, so the frame is padded on the
    right and bottom. Flow is cropped back to the original size.
    """
    # torchvision Normalize(mean=0.5, std=0.5): [0, 1] -> [-1, 1]
    image = image.float().mul(2).sub(1)
    _, height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    return image


def sample_flow(flow: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinear-sample ``flow`` ``[2, H, W]`` at pixel coordinates ``[N, 2]`` (x, y)."""
    _, height, width = flow.shape
    grid = torch.stack([
        (points[:, 0] + 0.5) / width,
        (points[:, 1] + 0.5) / height,
    ], dim=-1)
    grid = grid.mul(2).sub(1).view(1, 1, -1, 2)
    sampled = F.grid_sample(
        flow.float().unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return sampled[0, :, 0].transpose(0, 1)


class RAFTPointTracker(AbstractViewPointTracker):
    """Track queried points with torchvision RAFT optical flow.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]``, at least 128 pixels
    on each side, and share one resolution. Query points are ``[x, y]``
    pixels in their source frames.
    Each source frame is matched to every other frame with one RAFT forward,
    the same pair input as the torchvision optical-flow tutorial. The last
    flow iterate is ``[2, H, W]`` with channels ``(dx, dy)`` in source pixels.

    The source frame keeps the query coordinate. Predicted points are the
    query plus the sampled flow. RAFT has no occlusion or confidence head, so
    ``visibility`` and ``confidence`` are 1.
    """

    def __init__(self, model: str = "large", num_flow_updates: int = 12):
        if num_flow_updates <= 0:
            raise ValueError("num_flow_updates must be a positive integer")
        self.model = load_raft(model)
        self.num_flow_updates = num_flow_updates
        self.device = torch.device("cpu")
        self.model.eval()

    def to(self, device) -> 'RAFTPointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def track_batch(
            self,
            points: torch.Tensor,
            frame_indices: torch.Tensor,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[torch.Tensor | None]) -> Track:
        del frame_masks
        if any(frame.shape[0] != 3 for frame in frames):
            raise ValueError("RAFTPointTracker expects RGB frames with shape [3, H, W]")
        if any(frame.shape[-2:] != frames[0].shape[-2:] for frame in frames):
            raise ValueError("RAFTPointTracker expects all frames in a view to have the same resolution")
        if points.shape[0] == 0:
            raise ValueError("RAFTPointTracker expects at least one query point")
        if int(frame_indices.min()) < 0 or int(frame_indices.max()) >= len(frames):
            raise ValueError("frame_indices must point at frames in the sequence")

        height, width = frames[0].shape[-2:]
        # Downsampled by 8. The correlation pyramid needs a feature map of at least 16.
        padded_h = height + (8 - height % 8) % 8
        padded_w = width + (8 - width % 8) % 8
        if padded_h < 128 or padded_w < 128:
            raise ValueError("RAFTPointTracker expects frames of at least 128x128")
        images = [
            preprocess(frame.detach().clamp(0, 1)).to(self.device)
            for frame in frames
        ]
        query_points = points.to(device=self.device, dtype=torch.float32)
        frame_indices = frame_indices.to(device=self.device, dtype=torch.long)
        n_frames = len(images)
        pred = query_points.new_zeros((n_frames, query_points.shape[0], 2))
        visibility = query_points.new_ones((n_frames, query_points.shape[0]))
        confidence = query_points.new_ones((n_frames, query_points.shape[0]))

        for source_index in torch.unique(frame_indices).tolist():
            selected = frame_indices == source_index
            source_points = query_points[selected]
            source = images[source_index]
            for target_index, target in enumerate(images):
                if target_index == source_index:
                    pred[target_index, selected] = source_points
                    continue
                # Tutorial: list_of_flows = model(img1, img2); predicted = list_of_flows[-1]
                flows = self.model(
                    source.unsqueeze(0),
                    target.unsqueeze(0),
                    num_flow_updates=self.num_flow_updates,
                )
                flow = flows[-1][0, :, :height, :width]
                displacement = sample_flow(flow, source_points)
                pred[target_index, selected] = source_points + displacement.to(source_points.dtype)

        pred = pred.to(device=points.device, dtype=points.dtype)
        visibility = visibility.to(device=points.device, dtype=points.dtype)
        confidence = confidence.to(device=points.device, dtype=points.dtype)
        return Track(
            points=pred,
            visibility=visibility,
            confidence=confidence,
            mask=torch.ones(visibility.shape, dtype=torch.bool, device=pred.device),
        )
