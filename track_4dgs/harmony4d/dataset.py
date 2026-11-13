"""Read Harmony4D sequences as supervision samples."""

from argparse import ArgumentParser, Namespace

from ..dataset import AbstractGroupArgsDataset
from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import Harmony4DSpec


class Harmony4DDataset(AbstractLazyLoadDataset[Harmony4DSpec], AbstractGroupArgsDataset):
    """One sample is one Harmony4D sequence.

    Repeat ``--sequence`` for another sample. ``--frame_start``, ``--frame_end``,
    ``--frame_step``, and ``--init-dataset-index`` attach to the preceding
    ``--sequence``. Frames are ``range(frame_start, frame_end + 1, frame_step)``
    using 0-based frame indices. ``frame_start`` defaults to 0. ``frame_end``
    defaults to the last frame. ``frame_step`` defaults to 1.
    ``--init-dataset-index`` indexes that range and defaults to 0.

    Each sequence folder contains ``exo/``, ``ego/``, ``processed_data/``, and
    ``colmap/``. Views are the exo cameras. Tracks are the 17 COCO joints that
    were triangulated and manually checked, projected into each view.
    Construction records one :class:`Harmony4DSpec` per sequence.
    :meth:`Harmony4DSpec.load` reads the cameras and poses.
    """

    arg_group_starters = ("--sequence",)

    def __init__(self, *samples: Harmony4DSpec, device: str = "cpu"):
        if not samples:
            raise ValueError("at least one --sequence is required")
        super().__init__(*samples)
        self.device = device

    @staticmethod
    def add_group_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register one ``--sequence`` group's flags on ``parser`` and return ``parser``."""

        parser.add_argument("--sequence", required=True, type=str, help="Sequence folder for one sample.")
        parser.add_argument("--frame_start", type=int, default=0, help="First frame index for this sequence. Defaults to 0.")
        parser.add_argument("--frame_end", type=int, default=None, help="Last frame index for this sequence. Defaults to the last frame.")
        parser.add_argument("--frame_step", type=int, default=1, help="Frame step for this sequence. Defaults to 1.")
        parser.add_argument("--init-dataset-index", type=int, default=0, help="Index into range(frame_start, frame_end + 1, frame_step) for the query frame. Defaults to 0.")
        return parser

    @staticmethod
    def add_common_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register flags shared by every sequence and return ``parser``."""

        parser.add_argument("--device", default="cpu", type=str)
        return parser

    @staticmethod
    def spec_from_args(_common_args: Namespace, group_args: Namespace) -> Harmony4DSpec:
        """One :class:`Harmony4DSpec` for this ``--sequence`` group.

        Does not read cameras. :meth:`Harmony4DSpec.load` does that.
        """

        return Harmony4DSpec(
            sequence=group_args.sequence,
            frame_start=group_args.frame_start,
            frame_end=group_args.frame_end,
            frame_step=group_args.frame_step,
            init_dataset_index=group_args.init_dataset_index,
        )

    @classmethod
    def parse_known_group_chunk(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse one sequence folder and its frame range.

        Frames are ``range(frame_start, frame_end + 1, frame_step)``.
        ``frame_start`` defaults to 0. ``frame_end`` defaults to the last frame.
        ``frame_step`` defaults to 1. ``--init-dataset-index`` defaults to 0.
        """

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_group_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def parse_known_common_argv(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse the device used when a sample is loaded."""

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_common_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def from_args(cls, common_args: Namespace, *group_args: Namespace) -> "Harmony4DDataset":
        """One deferred sequence per ``--sequence``. Cameras load in :meth:`__getitem__`."""

        assert group_args, "at least one --sequence is required"
        samples = tuple(cls.spec_from_args(common_args, group) for group in group_args)
        return cls(*samples, device=common_args.device)
