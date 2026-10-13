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

    def dump(self, folder: str) -> None:
        """Save every sample to ``folder/supervision{i}`` via :meth:`Supervision.save`."""
        for i in range(len(self)):
            self[i].save(os.path.join(folder, f"supervision{i}"))
