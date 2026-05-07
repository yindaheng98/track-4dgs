import os
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from vggt.models.vggt import VGGT

from track_4dgs.tracker import AbstractPointTracker, Query, Track

RESOLUTION = 518


def compute_square_padding(H: int, W: int) -> tuple[int, int, int, int]:
    """Compute center padding `(left, right, top, bottom)` to make an image square."""
    max_dim = max(H, W)
    pad_top = (max_dim - H) // 2
    pad_left = (max_dim - W) // 2
    pad_bottom = max_dim - H - pad_top
    pad_right = max_dim - W - pad_left
    return pad_left, pad_right, pad_top, pad_bottom


def padding_square(img: torch.Tensor, target_resolution: int = 1024) -> torch.Tensor:
    """Center-pad to square + bicubic resize, matching the official VGGT
    ``load_and_preprocess_images_square``.

    Args:
        img: (C, H, W) tensor in [0, 1].
        target_resolution: output square size (default 1024).

    Returns:
        (C, target_resolution, target_resolution) tensor.
    """
    _, H, W = img.shape
    pad_left, pad_right, pad_top, pad_bottom = compute_square_padding(H, W)
    square = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return F.interpolate(
        square.unsqueeze(0),
        size=(target_resolution, target_resolution),
        mode='bicubic',
        align_corners=False,
        antialias=True,
    ).clamp_(0, 1).squeeze(0)


def points_to_square(points: torch.Tensor, H: int, W: int, square_size: int = RESOLUTION) -> torch.Tensor:
    """Map original image pixel coordinates into the padded square image."""
    pad_left, _, pad_top, _ = compute_square_padding(H, W)
    scale = square_size / max(H, W)
    offset = points.new_tensor([pad_left, pad_top])
    return (points + offset) * scale


def points_from_square(points: torch.Tensor, H: int, W: int, square_size: int = RESOLUTION) -> torch.Tensor:
    """Map padded square image coordinates back to original image pixels."""
    pad_left, _, pad_top, _ = compute_square_padding(H, W)
    scale = max(H, W) / square_size
    offset = points.new_tensor([pad_left, pad_top])
    return points * scale - offset


def load_vggt(checkpoint: str = "checkpoints/vggt_1B_commercial.pt") -> VGGT:
    if os.path.isfile(checkpoint):
        model = VGGT()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    return model


class VGGTPointTracker(AbstractPointTracker):
    """Track queried points with VGGT TrackHead.

    Frames are expected to be RGB ``[3, H, W]`` tensors in ``[0, 1]`` and query
    points use pixel coordinates ``[x, y]`` in the original frame resolution.
    """

    def __init__(
            self,
            checkpoint: str = "checkpoints/vggt_1B_commercial.pt",
            img_load_resolution: int = 1024,
            iters: int = 4):
        self.model = load_vggt(checkpoint)
        self.model.eval()
        self.img_load_resolution = img_load_resolution
        self.iters = iters
        self.device = torch.device("cpu")

    def to(self, device) -> 'VGGTPointTracker':
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model.eval()
        return self

    @torch.no_grad()
    def track(
            self,
            query: Query,
            frames: Sequence[torch.Tensor],
            frame_masks: Sequence[torch.Tensor | None]) -> Track:
        """Track query points through a sequence of images.

        Args:
            query: Query points on the first frame, in original image pixels.
            frames: Sequence of (C, H, W) tensors in [0, 1] range.

        Returns:
            Per-image tracked points and VGGT visibility scores.
        """
        if any(frame.shape[0] != 3 for frame in frames):
            raise ValueError("VGGTPointTracker expects RGB frames with shape [3, H, W]")
        assert torch.all(query.frame_indices == 0), "VGGT TrackHead expects query points from the first frame"
        images = frames

        # 1. Preprocess each image: center-pad + bicubic to img_load_resolution
        frames = []
        orig_sizes = []
        for img in images:
            frames.append(padding_square(img, self.img_load_resolution))
            orig_sizes.append(img.shape[1:])

        # 2. Bilinear down to 518 (matches run_VGGT), then feed to aggregator
        batch = torch.stack(frames)
        if batch.shape[-2:] != (RESOLUTION, RESOLUTION):
            batch = F.interpolate(batch, size=(RESOLUTION, RESOLUTION), mode='bilinear', align_corners=False)
        batch = batch.unsqueeze(0)
        device = batch.device
        dtype = (
            torch.bfloat16
            if device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = self.model.aggregator(batch)

        # 3. Predict per-image tracks from TrackHead
        H, W = orig_sizes[0]
        query_points = points_to_square(query.points, H, W).unsqueeze(0)
        with torch.cuda.amp.autocast(enabled=False):
            track_list, vis, _ = self.model.track_head(
                aggregated_tokens_list,
                images=batch,
                patch_start_idx=ps_idx,
                query_points=query_points,
                iters=self.iters,
            )
        points = track_list[-1].squeeze(0)
        visibility = vis.squeeze(0)

        # 4. Restore original pixel coordinates for each image
        track_points = []
        for i, (H, W) in enumerate(orig_sizes):
            track_points.append(points_from_square(points[i], H, W))
        track_points = torch.stack(track_points)

        return Track(points=track_points, visibility=visibility)
