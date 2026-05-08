import colorsys
import os
from collections.abc import Sequence

import torch
from PIL import Image, ImageDraw

from track_4dgs.registry import build_point_tracker, get_available_point_trackers
from track_4dgs.tracker import Query, Track


def load_image(path: str, device: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    data = data.view(image.height, image.width, 3)
    return data.permute(2, 0, 1).float().div(255).to(device)


def image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().clamp(0, 1).cpu()
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape [C, H, W]")
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    image = image[:3].permute(1, 2, 0)
    array = (image.numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def rainbow_colors(n: int) -> list[tuple[int, int, int]]:
    if n <= 0:
        return []
    return [
        tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0))
        for i in range(n)
    ]


def draw_rainbow_tracks(
    image: torch.Tensor,
    track: Track,
    frame_idx: int,
) -> Image.Image:
    canvas = image_tensor_to_pil(image)
    draw = ImageDraw.Draw(canvas)

    points = track.points.detach().cpu()
    visibility = track.visibility.detach().cpu().bool()
    colors = rainbow_colors(points.shape[1])

    for point_idx, color in enumerate(colors):
        visible = visibility[:frame_idx + 1, point_idx]
        if not visible.any():
            continue

        history_points = points[:frame_idx + 1, point_idx]
        last_visible_point = None
        for point, is_visible in zip(history_points, visible):
            xy = (float(point[0]), float(point[1]))
            if is_visible:
                if last_visible_point is not None:
                    draw.line([last_visible_point, xy], fill=color, width=1)
                last_visible_point = xy
            else:
                last_visible_point = None

        if visibility[frame_idx, point_idx]:
            x, y = points[frame_idx, point_idx].tolist()
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    return canvas


def sample_query(image: torch.Tensor, num_points: int) -> Query:
    _, height, width = image.shape
    points = torch.stack([
        torch.rand(num_points, device=image.device) * (width - 1),
        torch.rand(num_points, device=image.device) * (height - 1),
    ], dim=-1)
    frame_indices = torch.zeros(num_points, device=image.device, dtype=torch.long)
    return Query(points=points, frame_indices=frame_indices)


@torch.no_grad()
def rendering(frames: Sequence[torch.Tensor], track: Track, save: str) -> None:
    os.makedirs(save, exist_ok=True)
    print(f"Saving {track.points.shape[1]} tracked points to {save}")

    for frame_idx, frame in enumerate(frames):
        canvas = draw_rainbow_tracks(frame, track, frame_idx)
        canvas.save(os.path.join(save, f"{frame_idx:05d}.png"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("-s", "--sources", required=True, nargs="+", type=str, help="Source images in frame order.")
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--tracker", type=str, default="cotracker3", choices=get_available_point_trackers())
    parser.add_argument("-m", "--option_tracker", default=[], action="append", type=str)
    parser.add_argument("--num-points", default=256, type=int)
    args = parser.parse_args()
    tracker_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_tracker}

    with torch.no_grad():
        frames = [load_image(path, args.device) for path in args.sources]
        query = sample_query(frames[0], args.num_points)
        tracker = build_point_tracker(args.tracker, **tracker_configs).to(args.device)
        track = tracker(query, frames, [None] * len(frames))
        rendering(frames, track, args.destination)
