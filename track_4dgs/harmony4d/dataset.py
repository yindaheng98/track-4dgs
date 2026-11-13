"""Read Harmony4D sequences as supervision samples."""

import os
from argparse import ArgumentParser

from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import Harmony4DSpec


class Harmony4DDataset(AbstractLazyLoadDataset[Harmony4DSpec]):
    """One sample is one Harmony4D sequence.

    ``sequences`` lists sequence folders. Each folder contains ``exo/``,
    ``ego/``, ``processed_data/``, and ``colmap/``. Views are the exo cameras.
    Tracks are the 17 COCO joints that were triangulated and manually checked,
    projected into each view. ``start_index`` is the frame used as the query.
    Construction records one :class:`Harmony4DSpec` per sequence.
    :meth:`Harmony4DSpec.load` reads the cameras and poses.
    """

    def __init__(self, sequences: list[str], start_index: int = 0, device: str = "cpu"):
        if isinstance(sequences, str) or not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences):
            raise TypeError("sequences must be a list of str")
        if not sequences:
            raise ValueError("sequences must be non-empty")
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("start_index must be an int")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        for sequence in sequences:
            if not os.path.isdir(sequence):
                raise FileNotFoundError(sequence)
            for name in ("exo", "ego", "processed_data", "colmap"):
                path = os.path.join(sequence, name)
                if not os.path.isdir(path):
                    raise FileNotFoundError(path)
        super().__init__(*(
            Harmony4DSpec(sequence=sequence, start_index=start_index)
            for sequence in sequences
        ))
        self.device = device

    @classmethod
    def from_argv(cls, argv: list[str] = []) -> tuple["Harmony4DDataset", list[str]]:
        parser = ArgumentParser(allow_abbrev=False)
        parser.add_argument("--sequences", required=True, nargs="+", type=str)
        parser.add_argument("--start-index", default=0, type=int)
        parser.add_argument("--device", default="cpu", type=str)
        args, unknown = parser.parse_known_args(argv)
        return cls(args.sequences, start_index=args.start_index, device=args.device), unknown
