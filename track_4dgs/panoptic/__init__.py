from ..dataset.registry import register_supervision_dataset
from .dataset import PanopticDataset
from .spec import BODY_JOINT_COUNT, BODY_JOINT_NAMES, PanopticSpec

register_supervision_dataset("panoptic", PanopticDataset)

__all__ = ["BODY_JOINT_COUNT", "BODY_JOINT_NAMES", "PanopticDataset", "PanopticSpec"]
