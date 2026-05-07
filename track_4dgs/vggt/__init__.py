from track_4dgs.registry import register_point_tracker

from .tracker import VGGTPointTracker

register_point_tracker("vggt", VGGTPointTracker)

__all__ = [
    "VGGTPointTracker",
]
