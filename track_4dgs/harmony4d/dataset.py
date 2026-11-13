"""Read Harmony4D sequences as supervision samples."""

import os
from argparse import ArgumentParser

from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import Harmony4DSpec

_SKIP_DIRS = {".cache", "exo", "ego", "processed_data", "colmap"}


def discover_sequences(root: str) -> list[str]:
    sequences = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS]
        if (
            os.path.isdir(os.path.join(dirpath, "exo"))
            and os.path.isdir(os.path.join(dirpath, "processed_data"))
        ):
            sequences.append(dirpath)
    return sorted(sequences)


class Harmony4DDataset(AbstractLazyLoadDataset[Harmony4DSpec]):
    """One sample is one Harmony4D sequence.

    ``root`` contains ``train/`` and ``test/`` activity folders. Views are the
    ego RGB cameras followed by the exo cameras. Tracks are the 17 COCO
    joints that were triangulated and manually checked, projected into each
    view. ``start_index`` is the frame used as the query.
    Construction records one :class:`Harmony4DSpec` per sequence.
    :meth:`Harmony4DSpec.load` reads the cameras and poses.
    """

    def __init__(self, root: str, start_index: int = 0, device: str = "cpu"):
        if not os.path.isdir(root):
            raise FileNotFoundError(root)
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("start_index must be an int")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        sequences = discover_sequences(root)
        if not sequences:
            raise FileNotFoundError(f"no Harmony4D sequences under {root}")
        super().__init__(*(
            Harmony4DSpec(sequence=sequence, start_index=start_index)
            for sequence in sequences
        ))
        self.device = device

    @classmethod
    def from_argv(cls, argv: list[str] = []) -> tuple["Harmony4DDataset", list[str]]:
        parser = ArgumentParser(allow_abbrev=False)
        parser.add_argument("--root", required=True, type=str)
        parser.add_argument("--start-index", default=0, type=int)
        parser.add_argument("--device", default="cpu", type=str)
        args, unknown = parser.parse_known_args(argv)
        return cls(args.root, start_index=args.start_index, device=args.device), unknown
