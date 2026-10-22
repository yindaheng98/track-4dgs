import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, Optional, Protocol, TypeVar

from gaussian_splatting.dataset import CameraDataset
from torch.utils.data import Dataset

from ..tracker import AbstractPointTracker, Query
from .supervision import Supervision


class AbstractDataset(Dataset[Supervision], ABC):
    """Dataset of point-tracking :class:`~track_4dgs.dataset.supervision.Supervision` samples."""

    @abstractmethod
    def __getitem__(self, index: int) -> Supervision:
        """Return the supervision sample at ``index``."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def to(self, device) -> 'AbstractDataset':
        """Move samples this dataset yields onto ``device`` and return this dataset."""

        return self

    @classmethod
    @abstractmethod
    def from_argv(cls, argv: list[str] | None = None) -> tuple['AbstractDataset', list[str]]:
        """Parse ``argv`` and build this dataset.

        The second value is the unrecognized remainder of ``argv``.
        """
        raise NotImplementedError

    def dump(self, folder: str, start: int = 0) -> None:
        """Save sample ``i`` to ``folder/supervision{start + i}`` via :meth:`Supervision.save`."""
        if start < 0:
            raise ValueError("start must be non-negative")
        for i in range(len(self)):
            self[i].save(os.path.join(folder, f"supervision{start + i}"))


class PointSampleSpec(Protocol):
    """Deferred tracker input. Construction stores how to build the query and frames."""

    def load(self) -> tuple[Query, Sequence[CameraDataset]]:
        """Read this sample's cameras and return the query and frame-major datasets."""
        ...


PointSampleSpecType = TypeVar("PointSampleSpecType", bound=PointSampleSpec)


class AbstractPointTrackerDataset(AbstractDataset, Generic[PointSampleSpecType]):
    """Track each :class:`PointSampleSpec` into one :class:`Supervision`.

    ``tracker(query, frames, batch_size)`` supplies the tracks.
    ``batch_size`` is forwarded to every sample; ``None`` tracks all points at once.
    Specs stay unloaded until :meth:`__getitem__`.
    """

    def __init__(self, tracker: AbstractPointTracker, *samples: PointSampleSpecType, batch_size: Optional[int] = None):
        self.tracker = tracker
        self.samples = samples
        self.batch_size = batch_size
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
        tracks = self.tracker(query, frames, batch_size=self.batch_size)
        return Supervision(query=query, frames=frames, tracks=tracks)
