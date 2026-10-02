from track_4dgs.registry import register_point_tracker

from .tracker import VGGSfMPointTracker

register_point_tracker("vggsfm", VGGSfMPointTracker)

__all__ = [
    "VGGSfMPointTracker",
]
