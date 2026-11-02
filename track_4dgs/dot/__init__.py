from track_4dgs.registry import register_point_tracker

from .tracker import DOTPointTracker
from .dense import DOTDensePointTracker

register_point_tracker("dot", DOTPointTracker)

__all__ = [
    "DOTPointTracker",
    "DOTDensePointTracker",
]
