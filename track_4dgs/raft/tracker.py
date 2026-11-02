from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights, raft_large, raft_small

from track_4dgs.tracker import AbstractPairDensePointTracker
from track_4dgs.tracker.dense import sample_dense

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
    return sample_dense(flow, points)


class RAFTPointTracker(AbstractPairDensePointTracker):
    """Track queried points with torchvision RAFT optical flow.

    Frames are RGB ``[3, H, W]`` tensors in ``[0, 1]`` and share one
    resolution. Query points are ``[x, y]`` pixels in their source frames.
    A frame whose long side is already within ``max_resolution`` is not
    resized. A larger frame is shrunk so its long side is at most
    ``max_resolution``. ``track_pair_dense`` runs one RAFT forward and returns the
    last flow iterate scaled back onto the source frame. Channels are
    ``(dx, dy)``.

    RAFT has no occlusion or confidence head, so the dense result has no
    visibility or confidence map.
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
    def track_pair_dense(
            self,
            source: torch.Tensor,
            target: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if source.shape[-2:] != target.shape[-2:]:
            raise ValueError("RAFTPointTracker expects all frames in a view to have the same resolution")
        height, width = source.shape[-2:]
        new_h, new_w = flow_size(height, width, self.max_resolution)
        source_in = preprocess(source.detach().clamp(0, 1), (new_h, new_w)).to(self.device)
        target_in = preprocess(target.detach().clamp(0, 1), (new_h, new_w)).to(self.device)
        # Tutorial: list_of_flows = model(img1, img2); predicted = list_of_flows[-1]
        flows = self.model(
            source_in.unsqueeze(0),
            target_in.unsqueeze(0),
            num_flow_updates=self.num_flow_updates,
        )
        flow = flows[-1][0, :, :new_h, :new_w]
        flow = flow * flow.new_tensor([width / new_w, height / new_h]).view(2, 1, 1)
        if (new_h, new_w) != (height, width):
            flow = F.interpolate(
                flow.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False,
            ).squeeze(0)
        return flow, None, None
