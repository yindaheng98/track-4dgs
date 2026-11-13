from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from gaussian_splatting.dataset import CameraDataset

from ..tracker import Query, Track
from .dataset import AbstractDataset
from .supervision import Supervision


class SupervisionSpec(Protocol):
    """Deferred supervision sample. ``load`` returns the query, frames, and tracks."""

    def load(self, device) -> tuple[Query, Sequence[CameraDataset], Sequence[Track] | None]:
        """Read this sample on ``device``. ``tracks`` is ``None`` when the sample has no tracker."""
        ...


SupervisionSpecType = TypeVar("SupervisionSpecType", bound=SupervisionSpec)


class AbstractLazyLoadDataset(AbstractDataset, Generic[SupervisionSpecType]):
    """Load each :class:`SupervisionSpec` into one :class:`Supervision`.

    Specs stay unloaded until :meth:`__getitem__`.
    ``device`` stays ``None`` until :meth:`to`. The caller chooses it.
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
