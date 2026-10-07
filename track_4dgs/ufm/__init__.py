from track_4dgs.registry import register_point_tracker

from .tracker import UniFlowMatchPointTracker

register_point_tracker("uniflowmatch", UniFlowMatchPointTracker)

__all__ = [
    "UniFlowMatchPointTracker",
]
