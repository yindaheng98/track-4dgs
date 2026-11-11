from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from gaussian_splatting.dataset import CameraDataset

from ..tracker import Query, Track
from .dataset import AbstractDataset
from .supervision import Supervision


class SupervisionSpec(Protocol):
    """Deferred supervision sample. Construction stores how to read it."""

    def load(self, device) -> tuple[Query, Sequence[CameraDataset], Sequence[Track]]:
        """Read this sample on ``device`` and return the query, frames, and tracks."""
        ...


SupervisionSpecType = TypeVar("SupervisionSpecType", bound=SupervisionSpec)


class AbstractLazyLoadDataset(AbstractDataset, Generic[SupervisionSpecType]):
    """Load each :class:`SupervisionSpec` into one :class:`Supervision`.

    Specs stay unloaded until :meth:`__getitem__`.
    """

    def __init__(self, *samples: SupervisionSpecType):
        self.samples = samples
        self.device = None

    def __len__(self) -> int:
        return len(self.samples)

    def to(self, device) -> 'AbstractLazyLoadDataset':
        """Move samples this dataset yields onto ``device``."""

        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        query, frames, tracks = self.samples[index].load(self.device)
        return Supervision(query=query, frames=frames, tracks=tracks)
