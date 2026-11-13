"""Read DexYCB captures as supervision samples."""

import os
from argparse import ArgumentParser, BooleanOptionalAction

from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import DexYCBSpec


class DexYCBDataset(AbstractLazyLoadDataset[DexYCBSpec]):
    """One sample is one DexYCB capture.

    ``sequences`` lists capture folders. Each folder contains ``meta.yml``,
    color images, aligned depth, and joint labels.
    ``calibration_dir`` holds ``intrinsics/`` and ``extrinsics_*/``.
    Tracks are the 21 MANO joints in each color camera.
    ``start_index`` is the frame used as the query.
    ``load_depth`` reads the aligned depth with each color image.
    Construction records one :class:`DexYCBSpec` per capture.
    :meth:`DexYCBSpec.load` reads the cameras and joints.
    """

    def __init__(self, sequences: list[str], calibration_dir: str, start_index: int = 0, load_depth: bool = True, device: str = "cpu"):
        if isinstance(sequences, str) or not isinstance(sequences, list) or not all(isinstance(item, str) for item in sequences):
            raise TypeError("sequences must be a list of str")
        if not sequences:
            raise ValueError("sequences must be non-empty")
        if not os.path.isdir(calibration_dir):
            raise FileNotFoundError(calibration_dir)
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("start_index must be an int")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        for sequence in sequences:
            if not os.path.isdir(sequence):
                raise FileNotFoundError(sequence)
            meta = os.path.join(sequence, "meta.yml")
            if not os.path.isfile(meta):
                raise FileNotFoundError(meta)
        super().__init__(*(
            DexYCBSpec(
                sequence_dir=sequence,
                calibration_dir=calibration_dir,
                start_index=start_index,
                load_depth=load_depth,
            )
            for sequence in sequences
        ))
        self.device = device

    @classmethod
    def from_argv(cls, argv: list[str] = []) -> tuple["DexYCBDataset", list[str]]:
        parser = ArgumentParser(allow_abbrev=False)
        parser.add_argument("--sequences", required=True, nargs="+", type=str)
        parser.add_argument("--calibration-dir", required=True, type=str)
        parser.add_argument("--start-index", default=0, type=int)
        parser.add_argument("--load-depth", default=True, action=BooleanOptionalAction)
        parser.add_argument("--device", default="cpu", type=str)
        args, unknown = parser.parse_known_args(argv)
        return cls(
            args.sequences,
            args.calibration_dir,
            start_index=args.start_index,
            load_depth=args.load_depth,
            device=args.device,
        ), unknown
