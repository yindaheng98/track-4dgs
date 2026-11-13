from ..dataset.registry import register_supervision_dataset
from .dataset import DexYCBDataset
from .object import DexYCBObjectDataset, DexYCBObjectSpec
from .spec import DexYCBSpec
from .trajectory import query_and_tracks_from_trajectories

register_supervision_dataset("dexycb", DexYCBDataset)
register_supervision_dataset("dexycb-obj", DexYCBObjectDataset)

__all__ = [
    "DexYCBDataset",
    "DexYCBObjectDataset",
    "DexYCBObjectSpec",
    "DexYCBSpec",
    "query_and_tracks_from_trajectories",
]
