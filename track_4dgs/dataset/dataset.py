import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser

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
    def add_argument(cls, parser: ArgumentParser):
        """Register this dataset's arguments on ``parser``.

        Read ``--dataset`` before this runs, then parse in :meth:`from_args`::

            known, _ = parser.parse_known_args()
            cls = datasets[known.dataset]
            cls.add_argument(parser)
            dataset = cls.from_args(parser)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_args(cls, parser: ArgumentParser) -> 'AbstractDataset':
        """Parse arguments added by :meth:`add_argument` and build this dataset."""
        raise NotImplementedError

    def dump(self, folder: str) -> None:
        """Save every sample to ``folder/supervision{i}`` via :meth:`Supervision.save`."""
        for i in range(len(self)):
            self[i].save(os.path.join(folder, f"supervision{i}"))


class SavedDataset(AbstractDataset):
    """Load ``folder/supervision{i}`` samples written by :meth:`AbstractDataset.dump`."""

    def __init__(self, folder: str):
        self.folder = folder
        self.device = "cpu"
        index = 0
        while os.path.isdir(os.path.join(folder, f"supervision{index}")):
            index += 1
        self.length = index

    @classmethod
    def add_argument(cls, parser: ArgumentParser):
        parser.add_argument("--folder", required=True, type=str)

    @classmethod
    def from_args(cls, parser: ArgumentParser) -> 'SavedDataset':
        return cls(parser.parse_args().folder)

    def __len__(self) -> int:
        return self.length

    def to(self, device) -> 'SavedDataset':
        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        return Supervision.load(os.path.join(self.folder, f"supervision{index}"), device=self.device)
