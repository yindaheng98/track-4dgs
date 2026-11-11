"""The DexYCB views at one timestep.

Each frame is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
``len`` counts cameras. An image is read when that camera is indexed.
"""

import os

import torch
from gaussian_splatting.dataset import LazyCameraDataset

from .spec import DexYCBCameraSpec


class DexYCBFrame(LazyCameraDataset):
    """The DexYCB views at one timestep."""

    def __init__(
            self,
            sequence: str,
            serials: list[str],
            intrinsics: list[tuple[float, float, float, float]],
            rotations: list[torch.Tensor],
            translations: list[torch.Tensor],
            frame_index: int,
            device: str = "cpu",
    ):
        specs = []
        for serial, intrinsic, rotation, translation in zip(serials, intrinsics, rotations, translations):
            fx, fy, cx, cy = intrinsic
            folder = os.path.join(sequence, serial)
            specs.append(DexYCBCameraSpec(
                serial=serial,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                rotation=rotation,
                translation=translation,
                image_path=os.path.join(folder, f"color_{frame_index:06d}.jpg"),
                depth_path=os.path.join(folder, f"aligned_depth_to_color_{frame_index:06d}.png"),
            ))
        super().__init__(specs, load_mask=False, load_depth=True, device=device)
