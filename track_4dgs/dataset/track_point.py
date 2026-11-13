from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TypeVar

import torch
from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractPointTracker, Query, Track
from .lazy_load import AbstractLazyLoadDataset
from .supervision import Supervision


@dataclass(frozen=True, kw_only=True)
class PointSampleSpec(ABC):
    """Deferred tracker input.

    ``tracker`` stays ``None`` until a dataset copies one in with :func:`dataclasses.replace`.
    :meth:`load_query` reads the query and frames.
    :meth:`load` tracks them. ``tracks`` is ``None`` when ``tracker`` is ``None``.
    """

    tracker: AbstractPointTracker | None = None

    @abstractmethod
    def load_query(self, device) -> tuple[Query, Sequence[CameraDataset]]:
        """Read this sample on ``device`` and return the query and frame-major datasets."""

        raise NotImplementedError

    def load(self, device) -> tuple[Query, Sequence[CameraDataset], Sequence[Track] | None]:
        """Read this sample on ``device`` and track the query when ``tracker`` is set."""

        with torch.no_grad():
            query, frames = self.load_query(device)
            tracks = None if self.tracker is None else self.tracker(query, frames)
            return query, frames, tracks


PointSampleSpecType = TypeVar("PointSampleSpecType", bound=PointSampleSpec)


class AbstractPointTrackerDataset(AbstractLazyLoadDataset[PointSampleSpecType]):
    """Track each :class:`PointSampleSpec` into one :class:`Supervision`.

    :meth:`__getitem__` copies ``tracker`` onto the spec, then :meth:`PointSampleSpec.load` tracks the query.
    :meth:`to` moves that tracker and selects the device passed to :meth:`load`.
    """

    def __init__(self, tracker: AbstractPointTracker, *samples: PointSampleSpecType):
        super().__init__(*samples)
        self.tracker = tracker

    def to(self, device) -> 'AbstractPointTrackerDataset':
        """Move the tracker and select the device used by :meth:`PointSampleSpec.load`."""

        self.tracker = self.tracker.to(device)
        super().to(device)
        return self

    def __getitem__(self, index: int) -> Supervision:
        query, frames, tracks = replace(self.samples[index], tracker=self.tracker).load(self.device)
        return Supervision(query=query, frames=frames, tracks=tracks)
