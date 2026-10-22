"""Build supervision by tracking points sampled from a 3DGS model."""

import os
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from gaussian_splatting import GaussianModel
from gaussian_splatting.dataset import CameraDataset
from gaussian_splatting.prepare import prepare_gaussians

from ..tracker import Query
from .registry import register_supervision_dataset
from .tracker import SamplePointSpec, SamplePointTrackerDataset


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


@dataclass(frozen=True, kw_only=True)
class GaussianPointSampleSpec(SamplePointSpec):
    """Random Gaussian positions. Cameras still come from :meth:`SamplePointSpec.load_frames`."""

    sh_degree: int
    destination: str
    iteration: int

    def load(self) -> tuple[Query, list[CameraDataset]]:
        with torch.no_grad():
            frames = self.load_frames()
            load_ply = os.path.join(self.destination, "point_cloud", "iteration_" + str(self.iteration), "point_cloud.ply")
            gaussians = prepare_gaussians(
                sh_degree=self.sh_degree, source=self.sources[self.init_dataset_index], device=self.device,
                trainable_camera=False, load_ply=load_ply)
            query = query_views_from_gaussians(
                datasets=frames, gaussians=gaussians,
                init_dataset_index=self.init_dataset_index, num_points=self.num_points)
            return query, frames


class GaussianSplattingPointTrackerDataset(SamplePointTrackerDataset):
    """Like :class:`SamplePointTrackerDataset`, but each query is a random subset of a 3DGS point cloud.

    The checkpoint is read in :meth:`GaussianPointSampleSpec.load`, not at construction.
    """

    @staticmethod
    def add_group_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register one ``--sources`` group and the 3DGS checkpoint sampled for it."""
        SamplePointTrackerDataset.add_group_argument(parser)
        parser.add_argument("-d", "--destination", required=True, type=str, help="Gaussian run directory for this group.")
        parser.add_argument("-i", "--iteration", required=True, type=int, help="Checkpoint iteration for this group.")
        parser.add_argument("--sh_degree", default=3, type=int, help="Spherical-harmonics degree of this group's checkpoint. Defaults to 3.")
        return parser

    @staticmethod
    def spec_from_args(common_args: Namespace, group_args: Namespace) -> GaussianPointSampleSpec:
        """One :class:`GaussianPointSampleSpec` for this ``--sources`` group."""
        return GaussianPointSampleSpec(
            **vars(SamplePointTrackerDataset.spec_from_args(common_args, group_args)),
            sh_degree=group_args.sh_degree,
            destination=group_args.destination,
            iteration=group_args.iteration,
        )


register_supervision_dataset("tracker_gs", GaussianSplattingPointTrackerDataset)
