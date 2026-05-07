import colorsys
import os
from typing import Sequence, Tuple

import torch
from gaussian_splatting import Camera, GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_gaussians
from PIL import Image, ImageDraw

from track_4dgs.prepare import prepare_datasets, prepare_tracker
from track_4dgs.registry import get_available_point_trackers
from track_4dgs.tracker import Query, Track


def project_points(xyz: torch.Tensor, camera: Camera) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world points to camera pixel coordinates."""
    p_hom = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=xyz.device, dtype=xyz.dtype)], dim=1)
    p_hom = p_hom @ camera.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    p_proj = p_hom[:, :-1] / (p_hom[:, -1:] + 1e-7)

    width, height = camera.image_width, camera.image_height
    size = torch.tensor([[width, height]], device=xyz.device, dtype=xyz.dtype)
    pixels = (p_proj[:, :2] + 1.0) * size * 0.5 - 0.5
    valid_mask = (
        (p_hom[:, -1] > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels, valid_mask


def query_views_from_gaussians(
    datasets: Sequence[CameraDataset],
    gaussians: GaussianModel,
    init_dataset_index: int = 0,
    num_points: int = 256,
) -> list[Query]:
    n_views = len(datasets[init_dataset_index])
    xyz = gaussians.get_xyz.detach()
    order = torch.randperm(xyz.shape[0], device=xyz.device)
    queries = []
    for view_idx in range(n_views):
        pixels, valid_mask = project_points(xyz, datasets[init_dataset_index][view_idx])
        chosen = order[valid_mask[order]][:num_points]
        frame_indices = torch.full((chosen.numel(),), init_dataset_index, device=xyz.device, dtype=torch.long)
        queries.append(Query(points=pixels[chosen].float(), frame_indices=frame_indices))
    return queries


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


@torch.no_grad()
def rendering(datasets: Sequence[CameraDataset], save: str) -> None:
    os.makedirs(save, exist_ok=True)

    n_views = len(datasets[0])
    for view_idx in range(n_views):
        track = Track(
            points=torch.stack([dataset[view_idx].custom_data["track"].points for dataset in datasets]),
            visibility=torch.stack([dataset[view_idx].custom_data["track"].visibility for dataset in datasets]),
        )

        save_view = os.path.join(save, f"{view_idx:05d}")
        os.makedirs(save_view, exist_ok=True)
        print(f"Saving {track.points.shape[1]} tracked points in view {view_idx} to {save_view}")

        for frame_idx, dataset in enumerate(datasets):
            camera = dataset[view_idx]
            canvas = draw_rainbow_tracks(
                image=camera.ground_truth_image,
                track=track,
                frame_idx=frame_idx,
            )
            canvas.save(os.path.join(save_view, f"{frame_idx:05d}.png"))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-s", "--sources", required=True, nargs="+", type=str, help="Source frames for the sequence.")
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--load_cameras", default=None, nargs="+", type=str)
    parser.add_argument("--init-dataset-index", default=0, type=int, help="Index into --sources for the query frame.")
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--tracker", type=str, default="cotracker3", choices=get_available_point_trackers())
    parser.add_argument("-m", "--option_tracker", default=[], action="append", type=str)
    parser.add_argument("--num-points", default=256, type=int)
    args = parser.parse_args()
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration), f"track2d-{args.tracker}")
    tracker_configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_tracker}
    assert 0 <= args.init_dataset_index < len(args.sources), "--init-dataset-index must point to one of the values in --sources."
    if args.load_cameras is not None:
        assert len(args.load_cameras) == len(args.sources), "--load_cameras must have the same number of values as --sources."

    with torch.no_grad():
        datasets = prepare_datasets(
            sources=args.sources, device=args.device,
            trainable_camera=args.mode == "camera", load_cameras=args.load_cameras,
            load_mask=False, load_depth=False,
            reorder_reference_idx=args.init_dataset_index)
        gaussians = prepare_gaussians(
            sh_degree=args.sh_degree, source=args.sources[args.init_dataset_index], device=args.device,
            trainable_camera=args.mode == "camera", load_ply=load_ply)
        dataset_tracker = prepare_tracker(tracker_name=args.tracker, device=args.device, tracker_configs=tracker_configs)
        queries = query_views_from_gaussians(
            datasets=datasets, gaussians=gaussians,
            init_dataset_index=args.init_dataset_index, num_points=args.num_points)
        tracked_datasets = dataset_tracker(queries, datasets)

        rendering(
            datasets=tracked_datasets,
            save=save,
        )
