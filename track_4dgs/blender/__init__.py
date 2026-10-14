from ..dataset.registry import register_supervision_dataset
from ..registry import register_point_tracker
from .dataset import BlenderDataset
from .tracker import BlenderPointTracker

register_point_tracker("blender", BlenderPointTracker)
register_supervision_dataset("blender", BlenderDataset)

__all__ = [
    "BlenderPointTracker",
    "BlenderDataset",
]
