"""Read Panoptic sequences as supervision samples."""

from argparse import ArgumentParser, Namespace

from ..dataset import AbstractGroupArgsDataset
from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import PanopticSpec


class PanopticDataset(AbstractLazyLoadDataset[PanopticSpec], AbstractGroupArgsDataset):
    """One sample is one Panoptic sequence.

    Repeat ``--sequence`` for another sample. ``--frame_start``,
    ``--frame_end``, ``--frame_step``, ``--init-dataset-index``, and
    ``--view-indices`` attach to the preceding ``--sequence``.
    Frames are ``range(frame_start, frame_end + 1, frame_step)`` using HD frame indices.
    ``frame_start`` defaults to 0. ``frame_end`` defaults to the last body json.
    ``frame_step`` defaults to 1. ``--init-dataset-index`` indexes that range
    and defaults to 0. ``--view-indices`` indexes the HD cameras after they are
    sorted by name. Unset, it keeps every HD camera.

    Each sequence folder contains ``calibration_<name>.json``, body keypoints,
    and one HD video per camera. ``--num-points`` is the total number of tracks. The 19
    body joints of each person come first, and the rest are samples on a ribbon
    beside those bones. ``--max-image-size`` caps the long side of each image.
    Construction records one :class:`PanopticSpec` per sequence.
    :meth:`PanopticSpec.load` reads the cameras and joints.
    """

    arg_group_starters = ("--sequence",)

    def __init__(self, *samples: PanopticSpec):
        if not samples:
            raise ValueError("at least one --sequence is required")
        super().__init__(*samples)

    @staticmethod
    def add_group_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register one ``--sequence`` group's flags on ``parser`` and return ``parser``."""

        parser.add_argument("--sequence", required=True, type=str, help="Sequence folder for one sample.")
        parser.add_argument("--frame_start", type=int, default=0, help="First HD frame index for this sequence. Defaults to 0.")
        parser.add_argument("--frame_end", type=int, default=None, help="Last HD frame index for this sequence. Defaults to the last body json.")
        parser.add_argument("--frame_step", type=int, default=1, help="Frame step for this sequence. Defaults to 1.")
        parser.add_argument("--init-dataset-index", type=int, default=0, help="Index into range(frame_start, frame_end + 1, frame_step) for the query frame. Defaults to 0.")
        parser.add_argument("--view-indices", type=int, nargs="+", default=None, help="Indices into the HD cameras sorted by name. Defaults to every HD camera.")
        return parser

    @staticmethod
    def add_common_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register flags shared by every sequence and return ``parser``."""

        parser.add_argument("--num-points", default=256, type=int, help="Total points, including the body joints. Further points are sampled beside the bones. Defaults to 256.")
        parser.add_argument("--visible-cameras", default=2, type=int, help="Keep a surface point only when more cameras than this see it. Defaults to 2.")
        parser.add_argument("--inward-distance", default=0.0, type=float, help="How far surface points may move into the ribbon along its normal, in meters.")
        parser.add_argument("--outward-distance", default=0.0, type=float, help="How far surface points may move out of the ribbon along its normal, in meters.")
        parser.add_argument("--occlusion-eps", default=1e-2, type=float, help="Occlusion tolerance in meters. Defaults to 1e-2.")
        parser.add_argument("--max-image-size", default=None, type=int, help="Longest side of a loaded image. A larger photo is scaled down with its intrinsics. Defaults to the calibration resolution.")
        return parser

    @staticmethod
    def spec_from_args(common_args: Namespace, group_args: Namespace) -> PanopticSpec:
        """One :class:`PanopticSpec` for this ``--sequence`` group.

        Does not read cameras. :meth:`PanopticSpec.load` does that.
        """

        return PanopticSpec(
            sequence=group_args.sequence,
            frame_start=group_args.frame_start,
            frame_end=group_args.frame_end,
            frame_step=group_args.frame_step,
            init_dataset_index=group_args.init_dataset_index,
            view_indices=None if group_args.view_indices is None else tuple(group_args.view_indices),
            num_points=common_args.num_points,
            visible_cameras=common_args.visible_cameras,
            inward_distance=common_args.inward_distance,
            outward_distance=common_args.outward_distance,
            occlusion_eps=common_args.occlusion_eps,
            max_image_size=common_args.max_image_size,
        )

    @classmethod
    def parse_known_group_chunk(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse one sequence folder and its frame range.

        Frames are ``range(frame_start, frame_end + 1, frame_step)``.
        ``frame_start`` defaults to 0. ``frame_end`` defaults to the last body json.
        ``frame_step`` defaults to 1. ``--init-dataset-index`` defaults to 0.
        ``--view-indices`` defaults to every HD camera.
        """

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_group_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def parse_known_common_argv(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse flags shared by every sequence.

        ``--num-points`` defaults to 256. ``--max-image-size`` defaults to the
        calibration resolution.
        """

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_common_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def from_args(cls, common_args: Namespace, *group_args: Namespace) -> "PanopticDataset":
        """One deferred sequence per ``--sequence``. Cameras load in :meth:`__getitem__`."""

        assert group_args, "at least one --sequence is required"
        samples = tuple(cls.spec_from_args(common_args, group) for group in group_args)
        return cls(*samples)
