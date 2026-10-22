"""Supervision from points sampled uniformly in each source scene."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from ..registry import build_point_tracker, get_available_point_trackers
from ..tracker import AbstractPointTracker, Query
from ..utils.sample import sample_uniform_xyz
from .dataset import AbstractPointTrackerDataset
from .group_args import AbstractGroupArgsDataset
from .registry import register_supervision_dataset
from .reorder import prepare_datasets


@dataclass(frozen=True)
class SamplePointSpec:
    """Uniform scene-cube query. Cameras and the projection wait for :meth:`load`."""

    sources: tuple[str, ...]
    load_cameras: Optional[tuple[str, ...]]
    init_dataset_index: int
    num_points: int
    device: str
    visible_cameras: int = 2

    def load_frames(self) -> list[CameraDataset]:
        """Load this sample's cameras and align them to ``sources[init_dataset_index]``."""

        return prepare_datasets(
            sources=list(self.sources),
            device=self.device,
            trainable_camera=False,
            load_cameras=None if self.load_cameras is None else list(self.load_cameras),
            load_mask=False,
            load_depth=False,
            reorder_reference_idx=self.init_dataset_index,
        )

    def load(self) -> tuple[Query, list[CameraDataset]]:
        with torch.no_grad():
            frames = self.load_frames()
            xyz = sample_uniform_xyz(frames[self.init_dataset_index], self.num_points, self.visible_cameras, self.device)
            frame_indices = torch.full((self.num_points,), self.init_dataset_index, dtype=torch.long, device=self.device)
            return Query.from_projection(xyz, frames, frame_indices), frames


class SamplePointTrackerDataset(AbstractPointTrackerDataset[SamplePointSpec], AbstractGroupArgsDataset):
    """One uniformly sampled query per ``--sources`` group.

    :meth:`from_args` records specs. :meth:`SamplePointSpec.load` reads the cameras.
    """

    arg_group_starters = ("-s", "--sources")

    @staticmethod
    def add_group_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register one ``--sources`` group's flags on ``parser`` and return ``parser``."""
        parser.add_argument("-s", "--sources", required=True, nargs="+", type=str, metavar="SOURCE", help="Source frames for one sample.")
        parser.add_argument("--load_cameras", nargs="+", type=str, metavar="PATH", default=None, help="Camera file for each source in this group. Defaults to none.")
        parser.add_argument("--init-dataset-index", type=int, metavar="INDEX", default=0, help="Index into --sources for the query frame. Defaults to 0.")
        return parser

    @staticmethod
    def add_common_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register shared flags on ``parser`` and return ``parser``."""
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--tracker", type=str, default="cotracker3", choices=get_available_point_trackers())
        parser.add_argument("-m", "--option_tracker", default=[], action="append", type=str)
        parser.add_argument("--batch-size", default=None, type=int)
        parser.add_argument("--num-points", default=256, type=int)
        parser.add_argument("--visible-cameras", default=2, type=int, help="Keep a point only when more cameras than this see it.")
        return parser

    @staticmethod
    def tracker_from_args(common_args: Namespace) -> AbstractPointTracker:
        """Build the point tracker named by ``common_args``."""
        configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in common_args.option_tracker}
        return build_point_tracker(common_args.tracker, **configs).to(common_args.device)

    @staticmethod
    def specs_from_args(common_args: Namespace, *group_args: Namespace) -> tuple[SamplePointSpec, ...]:
        """One :class:`SamplePointSpec` per ``--sources`` group.

        Does not read cameras. :meth:`SamplePointSpec.load` does that.
        """
        assert group_args, "at least one --sources is required"
        samples = []
        for spec in group_args:
            sources = spec.sources
            cameras = spec.load_cameras
            init_dataset_index = spec.init_dataset_index
            assert 0 <= init_dataset_index < len(sources), "--init-dataset-index must point to one of the values in --sources."
            if cameras is not None:
                assert len(cameras) == len(sources), "--load_cameras must have the same number of values as --sources."
            samples.append(SamplePointSpec(
                sources=tuple(sources),
                load_cameras=None if cameras is None else tuple(cameras),
                init_dataset_index=init_dataset_index,
                num_points=common_args.num_points,
                visible_cameras=common_args.visible_cameras,
                device=common_args.device,
            ))
        return tuple(samples)

    @classmethod
    def parse_known_group_chunk(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse one ``--sources`` group.

        ``--load_cameras`` defaults to none. ``--init-dataset-index`` defaults to 0.
        """
        parser = ArgumentParser(allow_abbrev=False)
        cls.add_group_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def parse_known_common_argv(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse tracker inputs, how many points to draw, and the visible-camera limit."""
        parser = ArgumentParser(allow_abbrev=False)
        cls.add_common_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def from_args(cls, common_args: Namespace, *group_args: Namespace) -> 'SamplePointTrackerDataset':
        """One deferred uniform query per ``--sources``. Cameras load in :meth:`__getitem__`."""
        tracker = cls.tracker_from_args(common_args)
        samples = cls.specs_from_args(common_args, *group_args)
        return cls(tracker, *samples, batch_size=common_args.batch_size)


register_supervision_dataset("tracker", SamplePointTrackerDataset)
