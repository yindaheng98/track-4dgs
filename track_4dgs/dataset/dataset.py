import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

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
    def add_argument(cls, parser: ArgumentParser) -> None:
        """Register this dataset's arguments on ``parser``."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_namespace(cls, args: Namespace) -> 'AbstractDataset':
        """Build this dataset from a parsed argument namespace."""
        raise NotImplementedError

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> tuple['AbstractDataset', list[str]]:
        """Create a parser, parse ``argv``, and build this dataset.

        ``ArgumentParser.parse_known_args`` supplies :meth:`from_namespace`. The second
        value is the unrecognized remainder of ``argv``.
        """
        parser = ArgumentParser()
        cls.add_argument(parser)
        args, unknown = parser.parse_known_args(argv)
        return cls.from_namespace(args), unknown

    def dump(self, folder: str, start: int = 0) -> None:
        """Save sample ``i`` to ``folder/supervision{start + i}`` via :meth:`Supervision.save`."""
        if start < 0:
            raise ValueError("start must be non-negative")
        for i in range(len(self)):
            self[i].save(os.path.join(folder, f"supervision{start + i}"))


class SavedDataset(AbstractDataset):
    """Load ``folder/supervision{start + i}`` samples written by :meth:`AbstractDataset.dump`."""

    def __init__(self, folder: str, start: int = 0):
        if start < 0:
            raise ValueError("start must be non-negative")
        self.folder = folder
        self.start = start
        self.device = "cpu"
        index = start
        while os.path.isdir(os.path.join(folder, f"supervision{index}")):
            index += 1
        self.length = index - start

    @classmethod
    def add_argument(cls, parser: ArgumentParser) -> None:
        parser.add_argument("--folder", required=True, type=str)
        parser.add_argument("--start", type=int, default=0)

    @classmethod
    def from_namespace(cls, args: Namespace) -> 'SavedDataset':
        return cls(args.folder, args.start)

    def __len__(self) -> int:
        return self.length

    def to(self, device) -> 'SavedDataset':
        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        return Supervision.load(os.path.join(self.folder, f"supervision{self.start + index}"), device=self.device)
