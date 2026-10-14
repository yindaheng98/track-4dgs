"""Build supervision by running a point tracker."""

from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset
from tqdm import tqdm

from ..prepare import prepare_datasets
from ..registry import build_point_tracker, get_available_point_trackers
from ..tracker import AbstractPointTracker, Query
from ..utils.sample import sample_uniform_xyz
from .dataset import AbstractDataset
from .registry import register_supervision_dataset
from .supervision import Supervision


@dataclass(frozen=True)
class PointSample:
    """One tracker input: a view-major query and frame-major camera datasets."""

    query: Query
    frames: Sequence[CameraDataset]

    def to(self, device) -> 'PointSample':
        return PointSample(
            query=self.query.to(device),
            frames=tuple(dataset.to(device) for dataset in self.frames),
        )


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


class PointTrackerDataset(AbstractDataset):
    """Turn each :class:`PointTrackerSample` into one :class:`Supervision`.

    ``tracker(query, frames, batch_size)`` supplies the tracks.
    ``batch_size`` is forwarded to every sample; ``None`` tracks all points at once.
    """

    def __init__(self, tracker: AbstractPointTracker, *samples: PointSample, batch_size: Optional[int] = None):
        self.tracker = tracker
        self.samples = samples
        self.batch_size = batch_size

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
    def tracker_frames_from_namespace(args: Namespace) -> tuple[AbstractPointTracker, list[CameraDataset]]:
        """Build the tracker and one camera-dataset sequence per ``--sources`` group.

        Fills a missing ``init_dataset_index`` on each sample with 0.
        """
        assert args.samples, "at least one --sources is required"
        configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_tracker}
        tracker = build_point_tracker(args.tracker, **configs).to(args.device)
        frames: list[CameraDataset] = []
        for spec in args.samples:
            if spec.init_dataset_index is None:
                spec.init_dataset_index = 0
            sources = spec.sources
            cameras = spec.load_cameras
            init_dataset_index = spec.init_dataset_index
            assert 0 <= init_dataset_index < len(sources), "--init-dataset-index must point to one of the values in --sources."
            if cameras is not None:
                assert len(cameras) == len(sources), "--load_cameras must have the same number of values as --sources."
            frames.append(prepare_datasets(
                sources=sources, device=args.device,
                trainable_camera=False, load_cameras=cameras,
                load_mask=False, load_depth=False,
                reorder_reference_idx=init_dataset_index))
        return tracker, frames

    @classmethod
    def add_argument(cls, parser: ArgumentParser) -> None:
        """Register tracker inputs, how many points to draw, and the visible-camera limit."""
        cls.add_tracker_argument(parser)
        parser.add_argument("--num-points", default=256, type=int)
        parser.add_argument("--visible-cameras", default=2, type=int, help="Keep a point only when more cameras than this see it.")

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'PointTrackerDataset':
        """Project one uniformly sampled, sufficiently visible query per ``--sources``."""
        tracker, sample_frames = cls.tracker_frames_from_namespace(args)
        samples = []
        with torch.no_grad():
            for frames, spec in tqdm(
                    zip(sample_frames, args.samples),
                    total=len(sample_frames), desc="Samples", position=0):
                init_dataset_index = spec.init_dataset_index
                dataset = frames[init_dataset_index]
                xyz = sample_uniform_xyz(dataset, args.num_points, args.visible_cameras, args.device)
                frame_indices = torch.full((args.num_points,), init_dataset_index, dtype=torch.long, device=args.device)
                samples.append(PointSample(query=Query.from_projection(xyz, frames, frame_indices), frames=frames))
        return cls(tracker, *samples, batch_size=args.batch_size)

    def __len__(self) -> int:
        return len(self.samples)

    def to(self, device) -> 'PointTrackerDataset':
        self.tracker = self.tracker.to(device)
        self.samples = tuple(sample.to(device) for sample in self.samples)
        return self

    def __getitem__(self, index: int) -> Supervision:
        sample = self.samples[index]
        tracks = self.tracker(sample.query, sample.frames, batch_size=self.batch_size)
        return Supervision(query=sample.query, frames=sample.frames, tracks=tracks)


register_supervision_dataset("tracker", PointTrackerDataset)
