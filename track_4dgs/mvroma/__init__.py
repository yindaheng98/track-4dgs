from track_4dgs.registry import register_point_tracker

from .tracker import MVRoMaPointTracker

register_point_tracker("mvroma", MVRoMaPointTracker)

__all__ = [
    "MVRoMaPointTracker",
]
