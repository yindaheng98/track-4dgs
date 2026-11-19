"""Surface samples on ribbons beside the Panoptic COCO-19 bones."""

from collections.abc import Sequence

import torch
from gaussian_splatting.dataset import CameraDataset

from ..harmony4d.body import sample_posed_surface

RIBBON_RADIUS = 0.04
BONES = (
    (0, 1),
    (0, 2),
    (0, 3), (3, 4), (4, 5),
    (0, 9), (9, 10), (10, 11),
    (2, 6), (6, 7), (7, 8),
    (2, 12), (12, 13), (13, 14),
    (3, 9),
    (6, 12),
    (1, 15), (15, 16),
    (1, 17), (17, 18),
)


def bone_ribbons(joints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Triangles sitting 4 cm to one side of each bone.

    ``joints`` is ``(F, 19, 3)`` in meters. Each bone becomes two triangles
    between its ends and those ends shifted by :data:`RIBBON_RADIUS`. The
    shift is carried from the previous frame so the ribbon does not flip.
    A zero-length bone collapses and is not drawn. Returns vertices
    ``(F, V, 3)`` and faces ``(T, 3)`` int32.
    """

    if joints.ndim != 3 or joints.shape[1:] != (19, 3):
        raise ValueError(f"joints have shape {tuple(joints.shape)}, expected (frames, 19, 3)")
    starts = torch.tensor([bone[0] for bone in BONES], dtype=torch.long)
    ends = torch.tensor([bone[1] for bone in BONES], dtype=torch.long)
    origin = joints[:, starts]
    tip = joints[:, ends]
    direction = tip - origin
    length = torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-8)
    direction = direction / length
    side = torch.cross(direction[0], direction.new_tensor([0.0, 0.0, 1.0]).expand_as(direction[0]), dim=-1)
    fallback = torch.cross(direction[0], direction.new_tensor([0.0, 1.0, 0.0]).expand_as(direction[0]), dim=-1)
    side = torch.where(side.norm(dim=-1, keepdim=True) < 1e-6, fallback, side)
    side = side / side.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sides = [side]
    for frame_index in range(1, joints.shape[0]):
        carried = side - direction[frame_index] * (side * direction[frame_index]).sum(dim=-1, keepdim=True)
        carried_length = carried.norm(dim=-1, keepdim=True)
        replacement = torch.cross(
            direction[frame_index], direction.new_tensor([0.0, 1.0, 0.0]).expand_as(direction[frame_index]), dim=-1,
        )
        carried = torch.where(carried_length < 1e-6, replacement, carried)
        side = carried / carried.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        sides.append(side)
    offset = torch.stack(sides) * RIBBON_RADIUS
    vertices = torch.stack((origin, tip, origin + offset, tip + offset), dim=2)
    vertices = vertices.reshape(joints.shape[0], -1, 3)
    faces = []
    for index in range(len(BONES)):
        base = 4 * index
        faces.append((base, base + 2, base + 1))
        faces.append((base + 1, base + 2, base + 3))
    return vertices, torch.tensor(faces, dtype=torch.int32)


def sample_body_surface(
        joints: torch.Tensor,
        frames: Sequence[CameraDataset],
        init_dataset_index: int,
        num_points: int,
        visible_cameras: int,
        inward_distance: float,
        outward_distance: float,
        occlusion_eps: float,
        device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ``num_points`` bone-ribbon samples into every view.

    ``joints`` is ``(F, people, 19, 3)`` meters in the camera world, one row
    per selected frame. The ribbons are sampled and projected by
    :func:`~track_4dgs.harmony4d.body.sample_posed_surface`.
    Returns pixels ``(V, F, N, 2)`` and a bool mask ``(V, F, N)``.
    """

    if joints.ndim != 4 or joints.shape[2:] != (19, 3):
        raise ValueError(f"joints have shape {tuple(joints.shape)}, expected (frames, people, 19, 3)")
    mesh_device = torch.device("cpu" if device is None else device)
    posed = []
    mesh_faces = []
    for person in range(joints.shape[1]):
        vertices, faces = bone_ribbons(joints[:, person])
        posed.append(vertices.to(mesh_device))
        mesh_faces.append(faces.to(mesh_device))
    return sample_posed_surface(
        posed,
        mesh_faces,
        frames,
        init_dataset_index,
        num_points,
        visible_cameras,
        inward_distance,
        outward_distance,
        occlusion_eps,
    )
