from track_4dgs.registry import register_point_tracker

from .tracker import MatchaPointTracker

register_point_tracker("matcha", MatchaPointTracker)

__all__ = [
    "MatchaPointTracker",
]
