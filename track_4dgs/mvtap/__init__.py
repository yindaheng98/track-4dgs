from track_4dgs.registry import register_point_tracker

from .tracker import MVTAPPointTracker

register_point_tracker("mvtap", MVTAPPointTracker)

__all__ = [
    "MVTAPPointTracker",
]
