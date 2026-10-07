"""Bind triangulated query rays to rigid mesh objects."""

from collections import defaultdict
from collections.abc import Sequence

import torch
from gaussian_splatting.dataset import CameraDataset

from ...tracker import Query
from ..geometry import pack_object_faces, posed_scene_vertices, world_to_object
from .closest_point import closest_point
from .point_rays import PointRays
from .ray_cast import ray_cast
from .vote import ray_votes


def bind_points(
        verts: Sequence[torch.Tensor],
        faces: Sequence[torch.Tensor],
        query: Query,
        frames: Sequence[CameraDataset],
        matrices: torch.Tensor,
        device: str):
    """Vote query rays onto the posed mesh and snap each point into object space.

    ``verts`` and ``faces`` are per-object tensors in export order.
    ``matrices`` is ``(F, M, 4, 4)`` ``matrix_world``.
    ``device`` is the Warp device used for ray casts and closest-point snaps.
    A point with fewer than two in-image views stays masked off.
    Each ray votes for the object of its first hit, or for empty space on a miss.
    The winner must beat both empty space and the runner-up.
    A successful snap stores object-local coordinates, with confidence equal to the winning vote fraction.
    Otherwise the point keeps its triangulated world position at object index ``-1``.
    Confidence is then the fraction of rays that did not vote for the rejected winner.
    Returns ``obj_index`` ``(N,)`` long, ``local`` ``(N, 3)``, ``confidence`` ``(N,)``, and ``valid`` ``(N,)`` int.
    """
    n_points = query.points.shape[1]
    obj_index = torch.full((n_points,), -1, dtype=torch.long)
    local = torch.zeros((n_points, 3))
    confidence = torch.zeros((n_points,))
    valid = torch.zeros((n_points,), dtype=torch.int32)
    packed_faces, face_owner = pack_object_faces(verts, faces)
    vertex_starts = []
    vertex_start = 0
    for obj_verts in verts:
        vertex_starts.append(vertex_start)
        vertex_start += obj_verts.shape[0]
    frame = query.frame_indices[0].detach().cpu().to(dtype=torch.long)
    n_views = query.points.shape[0]
    for frame_index in frame.unique().tolist():
        selected = torch.where(frame == frame_index)[0]
        cameras = [frames[frame_index][view_index] for view_index in range(n_views)]
        rays = PointRays.from_cameras(query.points[:, selected], cameras, query.in_image[:, selected])
        local_ids = torch.where(rays.valid)[0]
        if local_ids.numel() == 0:
            continue
        view_mask = rays.ray_mask[local_ids]
        origins = rays.origins[local_ids][view_mask]
        directions = rays.directions[local_ids][view_mask]
        matrix_world = matrices[frame_index]
        world_verts = posed_scene_vertices(verts, matrix_world)
        hit_faces = ray_cast(origins, directions, world_verts, packed_faces, device)
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
                pending[winner].append((point_index, count, n_rays, rays.xyz[local_index]))
            else:
                local[point_index] = rays.xyz[local_index]
                confidence[point_index] = (n_rays - count) / n_rays
        for obj, items in pending.items():
            xyz = torch.stack([item[3] for item in items])
            start = vertex_starts[obj]
            obj_world_verts = world_verts[start:start + verts[obj].shape[0]]
            snapped_ok, snapped = closest_point(xyz, obj_world_verts, faces[obj], device)
            snapped = world_to_object(snapped, matrix_world[obj])
            for (point_index, count, n_rays, world_xyz), ok, snapped_xyz in zip(items, snapped_ok, snapped):
                if ok.item():
                    obj_index[point_index] = obj
                    local[point_index] = snapped_xyz
                    confidence[point_index] = count / n_rays
                else:
                    local[point_index] = world_xyz
                    confidence[point_index] = (n_rays - count) / n_rays
    return obj_index, local, confidence, valid
