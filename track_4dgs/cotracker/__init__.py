from track_4dgs.registry import register_point_tracker

from .tracker import Cotracker3PointTracker

register_point_tracker("cotracker3", Cotracker3PointTracker)

__all__ = [
    "Cotracker3PointTracker",
]
