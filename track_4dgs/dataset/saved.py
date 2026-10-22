"""Load dumped supervision folders."""

import os
from argparse import ArgumentParser

from .dataset import AbstractDataset
from .supervision import Supervision


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
    def from_argv(cls, argv: list[str] | None = None) -> tuple['SavedDataset', list[str]]:
        parser = ArgumentParser(allow_abbrev=False)
        parser.add_argument("--folder", required=True, type=str)
        parser.add_argument("--start", type=int, default=0)
        args, unknown = parser.parse_known_args(argv)
        return cls(args.folder, args.start), unknown

    def __len__(self) -> int:
        return self.length

    def to(self, device) -> 'SavedDataset':
        self.device = device
        return self

    def __getitem__(self, index: int) -> Supervision:
        return Supervision.load(os.path.join(self.folder, f"supervision{self.start + index}"), device=self.device)
