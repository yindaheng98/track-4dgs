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


def flow_size(height: int, width: int, max_resolution: int) -> tuple[int, int]:
    """Return the image size RAFT should see before the multiple-of-8 pad.

    A frame whose long side is already within ``max_resolution`` keeps its
    own size. A larger frame is shrunk so the long side is at most
    ``max_resolution``, rounding down to a multiple of 8. RAFT's feature
    encoder still only strides by 8, so this resize is what limits the
    correlation volume.
    """
    if max_resolution < 128:
        raise ValueError("max_resolution must be at least 128")
    if max(height, width) <= max_resolution:
        new_h, new_w = height, width
    elif height >= width:
        new_h = max_resolution // 8 * 8
        new_w = (width * new_h // height) // 8 * 8
    else:
        new_w = max_resolution // 8 * 8
        new_h = (height * new_w // width) // 8 * 8
    padded_h = new_h + (8 - new_h % 8) % 8
    padded_w = new_w + (8 - new_w % 8) % 8
    if padded_h < 128 or padded_w < 128:
        raise ValueError("RAFTPointTracker needs both sides at least 128")
    return new_h, new_w


def preprocess(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Map one RGB ``[3, H, W]`` frame in ``[0, 1]`` to a RAFT input.

    ``size`` is the content size. It is smaller than the frame only when the
    long side exceeds ``max_resolution``. Otherwise the frame is left as-is
    and padded on the right and bottom so H and W are divisible by 8.
    Values are mapped from ``[0, 1]`` to ``[-1, 1]``.
    """
    image = image.float()
    if image.shape[-2:] != size:
        image = F.interpolate(
            image.unsqueeze(0), size=size, mode="bilinear", align_corners=False, antialias=True,
        ).squeeze(0)
    _, height, width = image.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")
    return image.clamp(0, 1).mul(2).sub(1)


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

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]`` and share one
    resolution. Query points are ``[x, y]`` pixels in their source frames.
    A frame whose long side is already within ``max_resolution`` is not
    resized. A larger frame is shrunk so its long side is at most
    ``max_resolution``. Each source frame is then matched with one RAFT
    forward. The last flow iterate is sampled in that image and scaled back
    to source pixels. Channels are ``(dx, dy)``.

    The source frame keeps the query coordinate. Predicted points are the
    query plus the sampled flow. RAFT has no occlusion or confidence head, so
    ``visibility`` and ``confidence`` are 1.
    """

    def __init__(self, model: str = "large", num_flow_updates: int = 12, max_resolution: int = 512):
        if num_flow_updates <= 0:
            raise ValueError("num_flow_updates must be a positive integer")
        if max_resolution < 128:
            raise ValueError("max_resolution must be at least 128")
        self.model = load_raft(model)
        self.num_flow_updates = num_flow_updates
        self.max_resolution = max_resolution
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
        new_h, new_w = flow_size(height, width, self.max_resolution)
        images = [
            preprocess(frame.detach().clamp(0, 1), (new_h, new_w)).to(self.device)
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
                flow = flows[-1][0, :, :new_h, :new_w]
                resized_points = torch.stack([
                    (source_points[:, 0] + 0.5) * new_w / width - 0.5,
                    (source_points[:, 1] + 0.5) * new_h / height - 0.5,
                ], dim=-1)
                displacement = sample_flow(flow, resized_points)
                displacement = displacement * displacement.new_tensor([width / new_w, height / new_h])
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
