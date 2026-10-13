"""Build supervision by running a point tracker."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractPointTracker, Query
from .dataset import AbstractDataset
from .supervision import Supervision


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

    @classmethod
    def from_kwargs(cls, **kwargs) -> 'PointTrackerDataset':
        raise NotImplementedError

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
