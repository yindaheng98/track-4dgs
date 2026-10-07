from ..registry import register_point_tracker
from .tracker import BlenderPointTracker

register_point_tracker("blender", BlenderPointTracker)

__all__ = [
    "BlenderPointTracker",
]
