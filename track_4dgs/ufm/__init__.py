from track_4dgs.registry import register_point_tracker

from .tracker import UniFlowMatchPointTracker


class UFMBasePointTracker(UniFlowMatchPointTracker):
    """``infinity1096/UFM-Base``."""

    def __init__(self):
        super().__init__(model="base")


class UFMRefinePointTracker(UniFlowMatchPointTracker):
    """``infinity1096/UFM-Refine``."""

    def __init__(self):
        super().__init__(model="refine")


class UFMBase980PointTracker(UniFlowMatchPointTracker):
    """``infinity1096/UFM-Base-980``."""

    def __init__(self):
        super().__init__(model="base-980")


class UFMRefine980PointTracker(UniFlowMatchPointTracker):
    """``infinity1096/UFM-Refine-980``."""

    def __init__(self):
        super().__init__(model="refine-980")


class UFMBaseDinov2LPointTracker(UniFlowMatchPointTracker):
    """``infinity1096/UFM-Base-DINOv2L-init``."""

    def __init__(self):
        super().__init__(model="base-dinov2l-init")


class UFMBaseDinov2GPointTracker(UniFlowMatchPointTracker):
    """``infinity1096/UFM-Base-DINOv2G-init``."""

    def __init__(self):
        super().__init__(model="base-dinov2g-init")


register_point_tracker("ufm-base", UFMBasePointTracker)
register_point_tracker("ufm-refine", UFMRefinePointTracker)
register_point_tracker("ufm-base-980", UFMBase980PointTracker)
register_point_tracker("ufm-refine-980", UFMRefine980PointTracker)
register_point_tracker("ufm-base-dinov2l", UFMBaseDinov2LPointTracker)
register_point_tracker("ufm-base-dinov2g", UFMBaseDinov2GPointTracker)

__all__ = [
    "UniFlowMatchPointTracker",
    "UFMBasePointTracker",
    "UFMRefinePointTracker",
    "UFMBase980PointTracker",
    "UFMRefine980PointTracker",
    "UFMBaseDinov2LPointTracker",
    "UFMBaseDinov2GPointTracker",
]
