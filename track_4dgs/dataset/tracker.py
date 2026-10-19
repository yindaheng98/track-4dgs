"""Supervision from points sampled uniformly in each source scene."""

from argparse import Action, ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from ..registry import build_point_tracker, get_available_point_trackers
from ..tracker import AbstractPointTracker, Query
from ..utils.sample import sample_uniform_xyz
from .dataset import AbstractPointTrackerDataset
from .registry import register_supervision_dataset
from .reorder import prepare_datasets


class AppendPointSample(Action):
    """Attach ``--load_cameras`` and ``--init-dataset-index`` to the preceding ``--sources``.

    Each ``--sources`` starts one sample. Each of those two options may follow it
    once. A second copy, or either option before the first ``--sources``, is an error.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        samples = getattr(namespace, "samples", None)
        if samples is None:
            samples = []
            namespace.samples = samples
        if option_string in ("-s", "--sources"):
            samples.append(Namespace(sources=values, load_cameras=None, init_dataset_index=None))
            return
        if not samples:
            parser.error(f"{option_string} must follow --sources")
        sample = samples[-1]
        field = "load_cameras" if option_string == "--load_cameras" else "init_dataset_index"
        if getattr(sample, field) is not None:
            parser.error(f"{option_string} already set for this --sources")
        setattr(sample, field, values)


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


class SamplePointTrackerDataset(AbstractPointTrackerDataset[SamplePointSpec]):
    """One uniformly sampled query per ``--sources`` group.

    :meth:`from_namespace` records specs. :meth:`SamplePointSpec.load` reads the cameras.
    """

    @staticmethod
    def add_tracker_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register the tracker and camera flags shared by tracker datasets.

        ``--load_cameras`` and ``--init-dataset-index`` belong to the preceding ``--sources``.
        Each may appear once in that group. ``--init-dataset-index`` defaults to 0.
        """
        parser.add_argument("-s", "--sources", required=True, dest="samples", action=AppendPointSample, nargs="+", type=str, metavar="SOURCE", help="Source frames for one sample. Repeat to start another sample.")
        parser.add_argument("--load_cameras", dest="samples", action=AppendPointSample, nargs="+", type=str, metavar="PATH", help="Camera file for each source in the preceding --sources. At most once.")
        parser.add_argument("--init-dataset-index", dest="samples", action=AppendPointSample, type=int, metavar="INDEX", help="Index into the preceding --sources for the query frame. At most once. Defaults to 0.")
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--tracker", type=str, default="cotracker3", choices=get_available_point_trackers())
        parser.add_argument("-m", "--option_tracker", default=[], action="append", type=str)
        parser.add_argument("--batch-size", default=None, type=int)
        return parser

    @staticmethod
    def tracker_frames_from_namespace(args: Namespace) -> tuple[AbstractPointTracker, list[Namespace]]:
        """Build the tracker and normalize one ``--sources`` group per sample.

        Fills a missing ``init_dataset_index`` on each sample with 0.
        Does not read cameras. :meth:`SamplePointSpec.load` does that.
        """
        assert args.samples, "at least one --sources is required"
        configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_tracker}
        tracker = build_point_tracker(args.tracker, **configs).to(args.device)
        for spec in args.samples:
            if spec.init_dataset_index is None:
                spec.init_dataset_index = 0
            sources = spec.sources
            cameras = spec.load_cameras
            init_dataset_index = spec.init_dataset_index
            assert 0 <= init_dataset_index < len(sources), "--init-dataset-index must point to one of the values in --sources."
            if cameras is not None:
                assert len(cameras) == len(sources), "--load_cameras must have the same number of values as --sources."
        return tracker, list(args.samples)

    @classmethod
    def add_argument(cls, parser: ArgumentParser) -> None:
        """Register tracker inputs, how many points to draw, and the visible-camera limit."""
        cls.add_tracker_argument(parser)
        parser.add_argument("--num-points", default=256, type=int)
        parser.add_argument("--visible-cameras", default=2, type=int, help="Keep a point only when more cameras than this see it.")

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'SamplePointTrackerDataset':
        """One deferred uniform query per ``--sources``. Cameras load in :meth:`__getitem__`."""
        tracker, groups = cls.tracker_frames_from_namespace(args)
        samples = tuple(
            SamplePointSpec(
                sources=tuple(group.sources),
                load_cameras=None if group.load_cameras is None else tuple(group.load_cameras),
                init_dataset_index=group.init_dataset_index,
                num_points=args.num_points,
                visible_cameras=args.visible_cameras,
                device=args.device,
            )
            for group in groups
        )
        return cls(tracker, *samples, batch_size=args.batch_size)


register_supervision_dataset("tracker", SamplePointTrackerDataset)
