"""Bind triangulated query rays to rigid mesh objects."""

from collections import defaultdict
from collections.abc import Sequence

import torch
from gaussian_splatting.dataset import CameraDataset

from ...tracker import Query
from .closest_point import closest_point
from .point_rays import PointRays
from .ray_cast import ray_cast
from .vote import ray_votes


def bind_points(
        world_verts: torch.Tensor,
        scene_faces: torch.Tensor,
        face_owner: torch.Tensor,
        query: Query,
        frames: Sequence[CameraDataset],
        matrices: torch.Tensor,
        device: str):
    """Bind query rays and pose each point through every frame.

    ``world_verts`` is ``(F, P, 3)`` posed vertices, one row of frames.
    ``scene_faces`` is ``(T, 3)`` triangles into those vertices, and ``face_owner`` is ``(T,)``.
    ``matrices`` is ``(F, M, 4, 4)`` ``matrix_world``.
    ``device`` is the Warp device used for ray casts and closest-point snaps.
    A point with fewer than two in-image views stays masked off.
    Each ray votes for the object of its first hit, or for empty space on a miss.
    The winner must beat both empty space and the runner-up.
    A snap is multiplied by every frame's ``matrix_world``.
    A point that does not snap keeps its triangulated world position on every frame.
    Returns ``worlds`` ``(F, N, 3)`` and ``valid`` ``(N,)`` int.
    """
    n_points = query.points.shape[1]
    obj_index = torch.full((n_points,), -1, dtype=torch.long)
    local = torch.zeros((n_points, 3))
    valid = torch.zeros((n_points,), dtype=torch.int32)
    frame = query.frame_indices[0].detach().cpu().to(dtype=torch.long)
    n_views = query.points.shape[0]
    for frame_index in frame.unique().tolist():
        selected = torch.where(frame == frame_index)[0]
        cameras = [frames[frame_index][view_index] for view_index in range(n_views)]
        matrix_world = matrices[frame_index]
        posed_verts = world_verts[frame_index]
        rays = PointRays.from_cameras(query.points[:, selected], cameras, query.in_image[:, selected])
        local_ids = torch.where(rays.valid)[0]
        if local_ids.numel() == 0:
            continue
        view_mask = rays.ray_mask[local_ids]
        origins = rays.origins[local_ids][view_mask]
        directions = rays.directions[local_ids][view_mask]
        hit_faces = ray_cast(origins, directions, posed_verts, scene_faces, device)
        object_ids = torch.full((hit_faces.shape[0],), -1, dtype=torch.long)
        hit = hit_faces >= 0
        if face_owner.numel() > 0:
            object_ids[hit] = face_owner[hit_faces[hit].long()]
        cursor = 0
        pending = defaultdict(list)
        counts = view_mask.sum(dim=-1).tolist()
        for local_index, n_rays in zip(local_ids.tolist(), counts):
            point_index = int(selected[local_index])
            winner, count, second, misses = ray_votes(object_ids[cursor:cursor + n_rays])
            cursor += n_rays
            valid[point_index] = 1
            if winner >= 0 and count > misses and count > second:
                pending[winner].append((point_index, rays.xyz[local_index]))
            else:
                local[point_index] = rays.xyz[local_index]
        for obj, items in pending.items():
            xyz = torch.stack([item[1] for item in items])
            snapped_ok, snapped = closest_point(xyz, posed_verts, scene_faces[face_owner == obj], device)
            snapped = snapped.to(dtype=matrix_world.dtype)
            hom_snap = torch.cat([
                snapped,
                torch.ones(snapped.shape[0], 1, dtype=snapped.dtype, device=snapped.device),
            ], dim=1)
            snapped = torch.einsum("ij,bj->bi", torch.linalg.inv(matrix_world[obj]), hom_snap)[:, :3]
            for (point_index, world_xyz), ok, snapped_xyz in zip(items, snapped_ok, snapped):
                if ok.item():
                    obj_index[point_index] = obj
                    local[point_index] = snapped_xyz
                else:
                    local[point_index] = world_xyz
    hom = torch.cat([local, torch.ones(n_points, 1)], dim=1)
    worlds = hom[:, :3].unsqueeze(0).expand(matrices.shape[0], -1, -1).clone()
    bound = obj_index >= 0
    if bool(bound.any()):
        posed = torch.einsum("fbij,bj->fbi", matrices[:, obj_index[bound]], hom[bound])
        worlds[:, bound] = posed[..., :3]
    return worlds, valid
