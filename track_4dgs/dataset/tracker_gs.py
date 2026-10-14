"""Build supervision by tracking points sampled from a 3DGS model."""

import os
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence

import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_gaussians

from ..tracker import Query
from .tracker import PointTrackerDataset, PointTrackerSample


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


class GaussianSplattingPointTrackerDataset(PointTrackerDataset):
    """Like :class:`PointTrackerDataset`, but each query is a random subset of a 3DGS point cloud."""

    @classmethod
    def add_argument(cls, parser: ArgumentParser) -> None:
        """Register tracker inputs and the 3DGS checkpoint sampled into each query."""
        cls.add_tracker_argument(parser)
        parser.add_argument("--num-points", default=256, type=int)
        parser.add_argument("--sh_degree", default=3, type=int)
        parser.add_argument("-d", "--destination", required=True, action="append", type=str, help="Gaussian run directory for one sample. Repeat once per --sources.")
        parser.add_argument("-i", "--iteration", required=True, action="append", type=int, help="Checkpoint iteration for one sample. Repeat once per --sources.")

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'GaussianSplattingPointTrackerDataset':
        """Project a random subset of each sample's Gaussian positions."""
        assert len(args.destination) == len(args.samples), "--destination must appear once per --sources."
        assert len(args.iteration) == len(args.samples), "--iteration must appear once per --sources."
        tracker, sample_frames = cls.tracker_frames_from_namespace(args)
        samples = []
        with torch.no_grad():
            for spec, frames, destination, iteration in zip(
                    args.samples, sample_frames, args.destination, args.iteration):
                init_dataset_index = spec.init_dataset_index
                load_ply = os.path.join(destination, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
                gaussians = prepare_gaussians(
                    sh_degree=args.sh_degree, source=spec.sources[init_dataset_index], device=args.device,
                    trainable_camera=False, load_ply=load_ply)
                query = query_views_from_gaussians(
                    datasets=frames, gaussians=gaussians,
                    init_dataset_index=init_dataset_index, num_points=args.num_points)
                samples.append(PointTrackerSample(query=query, frames=frames))
        return cls(tracker, *samples, batch_size=args.batch_size)
