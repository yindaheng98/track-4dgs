"""Read DexYCB captures as supervision samples."""

from argparse import ArgumentParser, Namespace

from ..dataset import AbstractGroupArgsDataset
from ..dataset.lazy_load import AbstractLazyLoadDataset
from .spec import DexYCBSpec


class DexYCBDataset(AbstractLazyLoadDataset[DexYCBSpec], AbstractGroupArgsDataset):
    """One sample is one DexYCB capture.

    Repeat ``--sequence`` for another sample. ``--frame_start``, ``--frame_end``,
    ``--frame_step``, and ``--init-dataset-index`` attach to the preceding
    ``--sequence``. Frames are ``range(frame_start, frame_end + 1, frame_step)``
    using 0-based frame indices. ``frame_start`` defaults to 0. ``frame_end``
    defaults to the last frame. ``frame_step`` defaults to 1.
    ``--init-dataset-index`` indexes that range and defaults to 0.

    Each sequence folder contains ``meta.yml``, color images, aligned depth,
    and joint labels. ``calibration_dir`` holds ``intrinsics/`` and
    ``extrinsics_*/``. ``models_dir`` is the ``models/`` folder from ``models.tar.gz``.
    Tracks are the 21 MANO joints, then samples on the posed YCB meshes.
    ``load_depth`` reads the aligned depth with each color image and defaults to false.
    Construction records one :class:`DexYCBSpec` per capture.
    :meth:`DexYCBSpec.load` reads the cameras, joints, and meshes.
    """

    arg_group_starters = ("--sequence",)

    def __init__(self, *samples: DexYCBSpec):
        if not samples:
            raise ValueError("at least one --sequence is required")
        super().__init__(*samples)

    @staticmethod
    def add_group_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register one ``--sequence`` group's flags on ``parser`` and return ``parser``."""

        parser.add_argument("--sequence", required=True, type=str, help="Capture folder for one sample.")
        parser.add_argument("--frame_start", type=int, default=0, help="First frame index for this capture. Defaults to 0.")
        parser.add_argument("--frame_end", type=int, default=None, help="Last frame index for this capture. Defaults to the last frame.")
        parser.add_argument("--frame_step", type=int, default=1, help="Frame step for this capture. Defaults to 1.")
        parser.add_argument("--init-dataset-index", type=int, default=0, help="Index into range(frame_start, frame_end + 1, frame_step) for the query frame. Defaults to 0.")
        return parser

    @staticmethod
    def add_common_argument(parser: ArgumentParser) -> ArgumentParser:
        """Register the calibration, mesh, and sampling flags shared by every capture and return ``parser``."""

        parser.add_argument("--calibration-dir", required=True, type=str)
        parser.add_argument("--models-dir", required=True, type=str, help="Folder of YCB meshes. Each object has textured_simple.obj.")
        parser.add_argument("--load-depth", action="store_true")
        parser.add_argument("--num-points", default=256, type=int, help="Surface samples on the posed meshes, appended after the hand joints. Defaults to 256.")
        parser.add_argument("--visible-cameras", default=2, type=int, help="Keep a point only when more cameras than this see it. Defaults to 2.")
        parser.add_argument("--inward-distance", default=0.0, type=float, help="How far points may move into the mesh along the face normal, in world units.")
        parser.add_argument("--outward-distance", default=0.0, type=float, help="How far points may move out of the mesh along the face normal, in world units.")
        parser.add_argument("--occlusion-eps", default=1e-2, type=float, help="Occlusion depth tolerance in world units. Defaults to 1e-2.")
        return parser

    @staticmethod
    def spec_from_args(common_args: Namespace, group_args: Namespace) -> DexYCBSpec:
        """One :class:`DexYCBSpec` for this ``--sequence`` group.

        Does not read cameras. :meth:`DexYCBSpec.load` does that.
        """

        return DexYCBSpec(
            sequence_dir=group_args.sequence,
            calibration_dir=common_args.calibration_dir,
            models_dir=common_args.models_dir,
            frame_start=group_args.frame_start,
            frame_end=group_args.frame_end,
            frame_step=group_args.frame_step,
            init_dataset_index=group_args.init_dataset_index,
            load_depth=common_args.load_depth,
            num_points=common_args.num_points,
            visible_cameras=common_args.visible_cameras,
            inward_distance=common_args.inward_distance,
            outward_distance=common_args.outward_distance,
            occlusion_eps=common_args.occlusion_eps,
        )

    @classmethod
    def parse_known_group_chunk(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse one capture folder and its frame range.

        Frames are ``range(frame_start, frame_end + 1, frame_step)``.
        ``frame_start`` defaults to 0. ``frame_end`` defaults to the last frame.
        ``frame_step`` defaults to 1. ``--init-dataset-index`` defaults to 0.
        """

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_group_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def parse_known_common_argv(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse the calibration directory, YCB meshes, sampling flags, and whether to read depth."""

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_common_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def from_args(cls, common_args: Namespace, *group_args: Namespace) -> "DexYCBDataset":
        """One deferred capture per ``--sequence``. Cameras load in :meth:`__getitem__`."""

        assert group_args, "at least one --sequence is required"
        samples = tuple(cls.spec_from_args(common_args, group) for group in group_args)
        return cls(*samples)
