"""Build supervision by running a point tracker."""

from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset
from tqdm import tqdm

from ..prepare import prepare_datasets, prepare_tracker
from ..registry import get_available_point_trackers
from ..tracker import AbstractPointTracker, Query
from ..utils import project_points
from .dataset import AbstractDataset
from .supervision import Supervision


def sample_uniform_xyz(dataset: CameraDataset, num_points: int, visible_cameras: int, device: str) -> torch.Tensor:
    """Sample ``num_points`` uniformly in the cube of radius :meth:`CameraDataset.scene_extent`.

    The cube is centered on the mean camera center. A point is kept only when
    it projects into more cameras than ``visible_cameras``.
    """
    radius = float(dataset.scene_extent())
    n_views = len(dataset)
    center = torch.stack([
        dataset[view].camera_center.detach().to(device=device, dtype=torch.float32)
        for view in range(n_views)
    ]).mean(dim=0)
    lo = center - radius
    hi = center + radius
    if visible_cameras >= n_views:
        raise ValueError("--visible-cameras must be smaller than the number of cameras")
    picked = torch.empty((0, 3), device=device)
    with tqdm(total=num_points, desc="Points", position=1, leave=False) as points_bar:
        while picked.shape[0] < num_points:
            xyz = lo + torch.rand(num_points, 3, device=device) * (hi - lo)
            count = torch.zeros(num_points, dtype=torch.long, device=device)
            for view in range(n_views):
                _, in_frustum = project_points(xyz, dataset[view])
                count += in_frustum.to(dtype=torch.long)
            picked = torch.cat([picked, xyz[count > visible_cameras]], dim=0)
            points_bar.n = min(int(picked.shape[0]), num_points)
            points_bar.refresh()
    return picked[:num_points]


@dataclass(frozen=True)
class PointTrackerSample:
    """One tracker input: a view-major query and frame-major camera datasets."""

    query: Query
    frames: Sequence[CameraDataset]

    def to(self, device) -> 'PointTrackerSample':
        return PointTrackerSample(
            query=self.query.to(device),
            frames=tuple(dataset.to(device) for dataset in self.frames),
        )


class PointTrackerDataset(AbstractDataset):
    """Turn each :class:`PointTrackerSample` into one :class:`Supervision`.

    ``tracker(query, frames, batch_size)`` supplies the tracks.
    ``batch_size`` is forwarded to every sample; ``None`` tracks all points at once.
    """

    def __init__(self, tracker: AbstractPointTracker, *samples: PointTrackerSample, batch_size: Optional[int] = None):
        self.tracker = tracker
        self.samples = samples
        self.batch_size = batch_size

    @staticmethod
    def add_tracker_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register the tracker and camera flags shared by tracker datasets."""
        parser.add_argument("-s", "--sources", required=True, nargs="+", action="append", type=str, help="Source frames for one sample. Repeat for each sample.")
        parser.add_argument("--load_cameras", default=None, nargs="+", action="append", type=str)
        parser.add_argument("--init-dataset-indices", required=True, action="append", type=int, help="Index into one --sources group for the query frame. Repeat once per --sources.")
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--tracker", type=str, default="cotracker3", choices=get_available_point_trackers())
        parser.add_argument("-m", "--option_tracker", default=[], action="append", type=str)
        parser.add_argument("--batch-size", default=None, type=int)
        parser.add_argument("--cache", default=None, type=str, help="Directory of the point-track cache bank. Saved query points are reused; only new points are tracked.")
        return parser

    @staticmethod
    def tracker_frames_from_namespace(args: Namespace) -> tuple[AbstractPointTracker, list[CameraDataset]]:
        """Build the tracker and one camera-dataset sequence per ``--sources`` group."""
        load_cameras = args.load_cameras if args.load_cameras is not None else [None] * len(args.sources)
        assert len(load_cameras) == len(args.sources), "--load_cameras must appear once per --sources."
        assert len(args.init_dataset_indices) == len(args.sources), "--init-dataset-indices must appear once per --sources."
        configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option_tracker}
        tracker = prepare_tracker(tracker_name=args.tracker, device=args.device, tracker_configs=configs, cache_dir=args.cache)
        frames: list[CameraDataset] = []
        for sources, cameras, init_dataset_index in zip(args.sources, load_cameras, args.init_dataset_indices):
            assert 0 <= init_dataset_index < len(sources), "--init-dataset-indices must point to one of the values in --sources."
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
        parser.add_argument("--visible-cameras", default=0, type=int, help="Keep a point only when more cameras than this see it.")

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'PointTrackerDataset':
        """Project one uniformly sampled, sufficiently visible query per ``--sources``."""
        tracker, sample_frames = cls.tracker_frames_from_namespace(args)
        samples = []
        with torch.no_grad():
            for frames, init_dataset_index in tqdm(
                    zip(sample_frames, args.init_dataset_indices),
                    total=len(sample_frames), desc="Samples", position=0):
                dataset = frames[init_dataset_index]
                xyz = sample_uniform_xyz(dataset, args.num_points, args.visible_cameras, args.device)
                frame_indices = torch.full((args.num_points,), init_dataset_index, dtype=torch.long, device=args.device)
                samples.append(PointTrackerSample(query=Query.from_projection(xyz, frames, frame_indices), frames=frames))
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
