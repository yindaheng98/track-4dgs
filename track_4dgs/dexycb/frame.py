"""The DexYCB views at one timestep.

Each frame is a :class:`~gaussian_splatting.dataset.LazyCameraDataset`.
``len`` counts cameras. An image is read when that camera is indexed.
"""

import os
from dataclasses import dataclass

import torch
from gaussian_splatting.camera import Camera, build_camera
from gaussian_splatting.dataset import LazyCameraDataset
from gaussian_splatting.utils import focal2fov

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


@dataclass(frozen=True)
class DexYCBCameraSpec:
    """One DexYCB color camera at one timestep. ``load`` reads the image."""

    serial: str
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: torch.Tensor
    translation: torch.Tensor
    image_path: str
    depth_path: str

    def load(self, load_mask: bool, load_depth: bool, device) -> Camera:
        return build_camera(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            FoVx=focal2fov(self.fx, IMAGE_WIDTH),
            FoVy=focal2fov(self.fy, IMAGE_HEIGHT),
            R=self.rotation,
            T=self.translation,
            cx=self.cx,
            cy=self.cy,
            image_path=self.image_path,
            depth_path=self.depth_path if load_depth else None,
            device=device,
            custom_data={"serial": self.serial},
            resize_if_mismatch=None,
        )


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
