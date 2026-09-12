from track_4dgs.registry import register_point_tracker

from .tracker import D4RTPointTracker

register_point_tracker("d4rt", D4RTPointTracker)

__all__ = [
    "D4RTPointTracker",
]
