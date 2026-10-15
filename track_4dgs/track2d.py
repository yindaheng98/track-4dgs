import os
from typing import Sequence

import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_gaussians

from track_4dgs.dataset import CameraDatasetTracker, PointTrackerDataset
from track_4dgs.track1v import draw_rainbow_tracks
from track_4dgs.tracker import Query, Track


def query_views_from_gaussians(
    datasets: Sequence[CameraDataset],
    gaussians: GaussianModel,
    init_dataset_index: int = 0,
    num_points: int = 256,
) -> Query:
    xyz = gaussians.get_xyz.detach()
    chosen = torch.randperm(xyz.shape[0], device=xyz.device)[:num_points]
    frame_indices = torch.full((chosen.numel(),), init_dataset_index, device=xyz.device, dtype=torch.long)
    return Query.from_projection(xyz[chosen], datasets, frame_indices)


@torch.no_grad()
def rendering(datasets: Sequence[CameraDataset], save: str) -> None:
    os.makedirs(save, exist_ok=True)

    n_views = len(datasets[0])
    for view_idx in range(n_views):
        track = Track(
            points=torch.stack([dataset[view_idx].custom_data["track"].points for dataset in datasets]),
            visibility=torch.stack([dataset[view_idx].custom_data["track"].visibility for dataset in datasets]),
            confidence=torch.stack([dataset[view_idx].custom_data["track"].confidence for dataset in datasets]),
            mask=torch.stack([dataset[view_idx].custom_data["track"].mask for dataset in datasets]),
        )

        print(f"Saving {track.points.shape[1]} tracked points in view {view_idx} to {save}")

        for frame_idx, dataset in enumerate(datasets, start=1):
            camera = dataset[view_idx]
            canvas = draw_rainbow_tracks(
                image=camera.ground_truth_image,
                track=track,
                frame_idx=frame_idx - 1,
            )
            path = os.path.join(save, f"frame{frame_idx}", f"{view_idx}.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            canvas.save(path)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    PointTrackerDataset.add_tracker_argument(parser)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("-d", "--destination", required=True, type=str)
    parser.add_argument("-i", "--iteration", required=True, type=int)
    parser.add_argument("--mode", choices=["base", "camera"], default="base")
    parser.add_argument("--num-points", default=256, type=int)
    args = parser.parse_args()
    if len(args.samples) != 1:
        parser.error("track2d accepts a single --sources group")
    spec = args.samples[0]
    load_ply = os.path.join(args.destination, "point_cloud", "iteration_" + str(args.iteration), "point_cloud.ply")
    save = os.path.join(args.destination, "ours_{}".format(args.iteration), f"track2d-{args.tracker}")

    with torch.no_grad():
        tracker, sample_frames = PointTrackerDataset.tracker_frames_from_namespace(args)
        datasets = sample_frames[0]
        gaussians = prepare_gaussians(
            sh_degree=args.sh_degree, source=spec.sources[spec.init_dataset_index], device=args.device,
            trainable_camera=args.mode == "camera", load_ply=load_ply)
        dataset_tracker = CameraDatasetTracker(tracker)
        queries = query_views_from_gaussians(
            datasets=datasets, gaussians=gaussians,
            init_dataset_index=spec.init_dataset_index, num_points=args.num_points)
        tracked_datasets = dataset_tracker(queries, datasets, batch_size=args.batch_size)

        rendering(
            datasets=tracked_datasets,
            save=save,
        )
