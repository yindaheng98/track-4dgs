import os
from collections.abc import Sequence

import numpy as np
import torch
from matplotlib import colormaps

from track_4dgs.registry import build_point_tracker, get_available_point_trackers
from track_4dgs.track1v import image_tensor_to_pil, load_image
from track_4dgs.tracker import AbstractDensePointTracker, DenseTrack


def rainbow_colors(size: int) -> torch.Tensor:
    """Jet colors in flattened pixel order. See ``submodules/dot/demo.py``."""
    if size <= 0:
        return torch.zeros(0, 3)
    col_range = np.arange(size) / max(size - 1, 1)
    col = torch.from_numpy(colormaps["jet"](col_range)[..., :3]).float()
    return col.view(-1, 3)


def cardinal_neighbors(pos: torch.Tensor, col: torch.Tensor, eps: float = 0.01):
    """Spread one point onto its four neighboring pixels. See ``demo.py``."""
    pos_nw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor()], dim=-1)
    pos_sw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor() + 1], dim=-1)
    pos_ne = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor()], dim=-1)
    pos_se = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor() + 1], dim=-1)
    w_n = pos[:, 1].floor() + 1 - pos[:, 1] + eps
    w_s = pos[:, 1] - pos[:, 1].floor() + eps
    w_w = pos[:, 0].floor() + 1 - pos[:, 0] + eps
    w_e = pos[:, 0] - pos[:, 0].floor() + eps
    weights = torch.stack([w_n * w_w, w_s * w_w, w_n * w_e, w_s * w_e], dim=0)
    positions = torch.cat([pos_nw, pos_sw, pos_ne, pos_se], dim=0)
    colors = torch.cat([torch.cat([weight[:, None] * col, weight[:, None]], dim=-1) for weight in weights], dim=0)
    return positions, colors


def draw(pos: torch.Tensor, vis: torch.Tensor, col: torch.Tensor, height: int, width: int):
    """Paint visible points into an image and an alpha mask. See ``demo.py``."""
    frame = pos.new_zeros(height * width, 4)
    visible = vis.bool()
    pos, col = cardinal_neighbors(pos[visible], col[visible])
    inbound = (pos[:, 0] >= 0) & (pos[:, 0] <= width - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= height - 1)
    pos = pos[inbound].round().long()
    col = col[inbound]
    frame.scatter_add_(0, (pos[:, 1] * width + pos[:, 0]).view(-1, 1).expand(-1, 4), col)
    frame = frame.view(height, width, 4)
    rgb, alpha = frame[..., :3], frame[..., 3]
    nonzero = alpha > 0
    rgb[nonzero] /= alpha[nonzero][..., None]
    return rgb, nonzero[..., None].float()


def dense_to_tracks(dense: DenseTrack) -> torch.Tensor:
    """Convert flow to the demo layout ``[T, H, W, 3]`` of ``[x, y, visibility]``."""
    _, _, height, width = dense.flow.shape
    ys = torch.arange(height, device=dense.flow.device, dtype=dense.flow.dtype)
    xs = torch.arange(width, device=dense.flow.device, dtype=dense.flow.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    visibility = dense.flow.new_ones(dense.flow.shape[0], height, width) if dense.visibility is None else dense.visibility
    return torch.stack([grid_x + dense.flow[:, 0], grid_y + dense.flow[:, 1], visibility], dim=-1)


def plot_overlay(frames: Sequence[torch.Tensor], tracks: torch.Tensor, overlay_factor: float = 0.75) -> list[torch.Tensor]:
    """Overlay rainbow correspondences, using the demo's left-to-right colors."""
    time, height, width, _ = tracks.shape
    flat = tracks.permute(0, 2, 1, 3).reshape(time, -1, 3)
    colors = rainbow_colors(flat.shape[1]).to(flat.device)
    video = []
    for index, frame in enumerate(frames):
        image = frame.detach().clamp(0, 1).to(flat.device).permute(1, 2, 0)
        rainbow, alpha = draw(flat[index, :, :2], flat[index, :, 2], colors, height, width)
        image = overlay_factor * alpha * rainbow + (1 - overlay_factor * alpha) * image
        video.append(image.permute(2, 0, 1))
    return video


@torch.no_grad()
def rendering(frames: Sequence[torch.Tensor], dense: DenseTrack, save: str, overlay_factor: float = 0.75) -> None:
    os.makedirs(save, exist_ok=True)
    print(f"Saving dense tracks of shape {tuple(dense.flow.shape)} to {save}")
    for frame_idx, frame in enumerate(plot_overlay(frames, dense_to_tracks(dense), overlay_factor)):
        image_tensor_to_pil(frame).save(os.path.join(save, f"{frame_idx + 1:05d}.png"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--sources", required=True, nargs="+", type=str, help="Source images in frame order.")
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--tracker", type=str, default="raft", choices=get_available_point_trackers())
    parser.add_argument("-m", "--option_tracker", default=[], action="append", type=str)
    parser.add_argument("--source-index", default=0, type=int)
    parser.add_argument("--overlay-factor", default=0.75, type=float)
    args = parser.parse_args()
    tracker_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_tracker}

    with torch.no_grad():
        frames = [load_image(path, args.device) for path in args.sources]
        if args.source_index < 0 or args.source_index >= len(frames):
            raise ValueError("source_index must point at a source image")
        tracker = build_point_tracker(args.tracker, **tracker_configs).to(args.device)
        if not isinstance(tracker, AbstractDensePointTracker):
            raise TypeError("track_dense requires an AbstractDensePointTracker")
        dense = tracker.track_dense(frames, [None] * len(frames), args.source_index)
        rendering(frames, dense, args.destination, overlay_factor=args.overlay_factor)
