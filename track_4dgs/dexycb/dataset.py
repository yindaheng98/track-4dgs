"""Read DexYCB captures as supervision samples."""

import os
from argparse import ArgumentParser, BooleanOptionalAction

from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import DexYCBSpec


def discover_sequences(sequence_dir: str) -> list[str]:
    sequences = []
    for subject in sorted(os.listdir(sequence_dir)):
        subject_dir = os.path.join(sequence_dir, subject)
        if not os.path.isdir(subject_dir):
            continue
        for name in sorted(os.listdir(subject_dir)):
            sequence = os.path.join(subject_dir, name)
            if os.path.isfile(os.path.join(sequence, "meta.yml")):
                sequences.append(sequence)
    return sequences


class DexYCBDataset(AbstractLazyLoadDataset[DexYCBSpec]):
    """One sample is one DexYCB capture.

    ``sequence_dir`` holds the subject folders. Each capture contains
    ``meta.yml``, color images, aligned depth, and joint labels.
    ``calibration_dir`` holds ``intrinsics/`` and ``extrinsics_*/``.
    Tracks are the 21 MANO joints in each color camera.
    ``start_index`` is the frame used as the query.
    ``load_depth`` reads the aligned depth with each color image.
    Construction records one :class:`DexYCBSpec` per capture.
    :meth:`DexYCBSpec.load` reads the cameras and joints.
    """

    def __init__(self, sequence_dir: str, calibration_dir: str, start_index: int = 0, load_depth: bool = True, device: str = "cpu"):
        if not os.path.isdir(sequence_dir):
            raise FileNotFoundError(sequence_dir)
        if not os.path.isdir(calibration_dir):
            raise FileNotFoundError(calibration_dir)
        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("start_index must be an int")
        if start_index < 0:
            raise ValueError("start_index must be non-negative")
        sequences = discover_sequences(sequence_dir)
        if not sequences:
            raise FileNotFoundError(f"no DexYCB sequences under {sequence_dir}")
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
        parser.add_argument("--sequence-dir", required=True, type=str)
        parser.add_argument("--calibration-dir", required=True, type=str)
        parser.add_argument("--start-index", default=0, type=int)
        parser.add_argument("--load-depth", default=True, action=BooleanOptionalAction)
        parser.add_argument("--device", default="cpu", type=str)
        args, unknown = parser.parse_known_args(argv)
        return cls(
            args.sequence_dir,
            args.calibration_dir,
            start_index=args.start_index,
            load_depth=args.load_depth,
            device=args.device,
        ), unknown
