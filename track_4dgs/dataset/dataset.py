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
    def from_kwargs(cls, **kwargs) -> 'AbstractDataset':
        """Build a dataset from keyword arguments, separately from ``__init__``.

        The only input is ``kwargs``. Callers can turn CLI options into that dict
        and unpack it::

            configs = {o.split("=", 1)[0]: eval(o.split("=", 1)[1]) for o in args.option}
            dataset = DatasetClass.from_kwargs(**configs)
        """
        return cls(**kwargs)

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

    def __len__(self) -> int:
        return self.length

    def to(self, device) -> 'SavedDataset':
        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        return Supervision.load(os.path.join(self.folder, f"supervision{index}"), device=self.device)
