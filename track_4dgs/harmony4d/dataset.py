"""Read Harmony4D sequences as supervision samples."""

from argparse import ArgumentParser, Namespace

from ..dataset import AbstractGroupArgsDataset
from ..dataset.lazy_load import AbstractLazyLoadDataset
from .body import DEFAULT_SMPL_FACES, DEFAULT_SMPL_FACES_URL
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
    ``colmap/``. Views are the exo cameras. ``--num-points`` is the total
    number of tracks. The checked COCO joints come first, and the rest are
    samples on the posed SMPL meshes. ``--max-image-size`` caps the long side
    of each image. A larger photo is scaled down with its intrinsics.
    Construction records one :class:`Harmony4DSpec` per sequence.
    :meth:`Harmony4DSpec.load` reads the cameras, poses, and meshes.
    """

    arg_group_starters = ("--sequence",)

    def __init__(self, *samples: Harmony4DSpec):
        if not samples:
            raise ValueError("at least one --sequence is required")
        super().__init__(*samples)

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

        parser.add_argument("--num-points", default=256, type=int, help="Total points, including the checked joints. Further points are sampled on the body. Defaults to 256.")
        parser.add_argument("--smpl-faces", default=DEFAULT_SMPL_FACES, type=str, help="Local SMPL triangle file. A file that cannot be loaded is replaced from --smpl-faces-url.")
        parser.add_argument("--smpl-faces-url", default=DEFAULT_SMPL_FACES_URL, type=str, help="Downloaded into --smpl-faces when that file cannot be loaded. Defaults to src/tf_smpl/smpl_faces.npy from akanazawa/hmr.")
        parser.add_argument("--visible-cameras", default=2, type=int, help="Keep a surface point only when more cameras than this see it. Defaults to 2.")
        parser.add_argument("--inward-distance", default=0.0, type=float, help="How far surface points may move into the mesh along the face normal, in the COLMAP world.")
        parser.add_argument("--outward-distance", default=0.0, type=float, help="How far surface points may move out of the mesh along the face normal, in the COLMAP world.")
        parser.add_argument("--occlusion-eps", default=1e-2, type=float, help="Occlusion tolerance in pose-world meters. Scaled into the COLMAP world. Defaults to 1e-2.")
        parser.add_argument("--max-image-size", default=None, type=int, help="Longest side of a loaded image. A larger photo is scaled down with its intrinsics. Defaults to the original size.")
        return parser

    @staticmethod
    def spec_from_args(common_args: Namespace, group_args: Namespace) -> Harmony4DSpec:
        """One :class:`Harmony4DSpec` for this ``--sequence`` group.

        Does not read cameras. :meth:`Harmony4DSpec.load` does that.
        """

        return Harmony4DSpec(
            sequence=group_args.sequence,
            frame_start=group_args.frame_start,
            frame_end=group_args.frame_end,
            frame_step=group_args.frame_step,
            init_dataset_index=group_args.init_dataset_index,
            num_points=common_args.num_points,
            smpl_faces=common_args.smpl_faces,
            smpl_faces_url=common_args.smpl_faces_url,
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
        ``frame_start`` defaults to 0. ``frame_end`` defaults to the last frame.
        ``frame_step`` defaults to 1. ``--init-dataset-index`` defaults to 0.
        """

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_group_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def parse_known_common_argv(cls, argv: list[str] = []) -> tuple[Namespace, list[str]]:
        """Parse flags shared by every sequence.

        ``--num-points`` defaults to 256. ``--smpl-faces`` and ``--smpl-faces-url``
        name the local triangle file and the URL used when that file cannot be loaded.
        ``--max-image-size`` defaults to the original image size.
        """

        parser = ArgumentParser(allow_abbrev=False)
        cls.add_common_argument(parser)
        return parser.parse_known_args(argv)

    @classmethod
    def from_args(cls, common_args: Namespace, *group_args: Namespace) -> "Harmony4DDataset":
        """One deferred sequence per ``--sequence``. Cameras load in :meth:`__getitem__`."""

        assert group_args, "at least one --sequence is required"
        samples = tuple(cls.spec_from_args(common_args, group) for group in group_args)
        return cls(*samples)
