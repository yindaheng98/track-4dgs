import os
from abc import ABC, abstractmethod

from torch.utils.data import Dataset

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
    def from_argv(cls, argv: list[str] = []) -> tuple['AbstractDataset', list[str]]:
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
