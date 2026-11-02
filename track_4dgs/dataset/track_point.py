from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractPointTracker, Query
from .dataset import AbstractDataset
from .supervision import Supervision


class PointSampleSpec(Protocol):
    """Deferred tracker input. Construction stores how to build the query and frames."""

    def load(self) -> tuple[Query, Sequence[CameraDataset]]:
        """Read this sample's cameras and return the query and frame-major datasets."""
        ...


PointSampleSpecType = TypeVar("PointSampleSpecType", bound=PointSampleSpec)


class AbstractPointTrackerDataset(AbstractDataset, Generic[PointSampleSpecType]):
    """Track each :class:`PointSampleSpec` into one :class:`Supervision`.

    ``tracker(query, frames)`` supplies the tracks.
    Specs stay unloaded until :meth:`__getitem__`.
    """

    def __init__(self, tracker: AbstractPointTracker, *samples: PointSampleSpecType):
        self.tracker = tracker
        self.samples = samples
        self.device = None

    def __len__(self) -> int:
        return len(self.samples)

    def to(self, device) -> 'AbstractPointTrackerDataset':
        """Move the tracker now. Loaded samples follow in :meth:`__getitem__`."""

        self.tracker = self.tracker.to(device)
        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        query, frames = self.samples[index].load()
        if self.device is not None:
            query = query.to(self.device)
            frames = tuple(dataset.to(self.device) for dataset in frames)
        tracks = self.tracker(query, frames)
        return Supervision(query=query, frames=frames, tracks=tracks)
