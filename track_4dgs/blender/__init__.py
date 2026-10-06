from track_4dgs.registry import register_point_tracker

from .dataset import BlenderPointTracker

register_point_tracker("blender", BlenderPointTracker)

__all__ = [
    "BlenderPointTracker",
]
