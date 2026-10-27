"""Bind triangulated query rays to deformed mesh surfaces."""

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
        device: str):
    """Bind query rays and replay each snap on the deformed surface.

    ``world_verts`` is ``(F, P, 3)`` posed vertices, one row of frames.
    ``scene_faces`` is ``(T, 3)`` triangles into those vertices, and ``face_owner`` is ``(T,)``.
    Vertex count and triangle indices are the same on every frame.
    ``device`` is the Warp device used for ray casts and closest-point snaps.
    A point with fewer than two in-image views stays masked off.
    Each ray votes for the object of its first hit, or for empty space on a miss.
    The winner must beat both empty space and the runner-up.
    A snap stores Warp barycentric weights on the winning object's triangle.
    Those weights are ``u`` on the first vertex, ``v`` on the second, and ``1 - u - v`` on the third.
    Every frame interpolates the same triangle from that frame's vertices.
    A point that does not snap keeps its triangulated world position on every frame.
    Returns ``worlds`` ``(F, N, 3)`` and ``valid`` ``(N,)`` int.
    """
    n_points = query.points.shape[1]
    n_frames = world_verts.shape[0]
    triangles = torch.full((n_points, 3), -1, dtype=torch.long, device=world_verts.device)
    bary = torch.zeros((n_points, 3), dtype=world_verts.dtype, device=world_verts.device)
    anchor = torch.zeros((n_points, 3), dtype=world_verts.dtype, device=world_verts.device)
    valid = torch.zeros((n_points,), dtype=torch.int32, device=world_verts.device)
    frame = query.frame_indices[0].detach().to(dtype=torch.long)
    n_views = query.points.shape[0]
    for frame_index in frame.unique().tolist():
        selected = torch.where(frame == frame_index)[0]
        cameras = [frames[frame_index][view_index] for view_index in range(n_views)]
        posed_verts = world_verts[frame_index]
        rays = PointRays.from_cameras(query.points[:, selected], cameras, query.in_image[:, selected])
        local_ids = torch.where(rays.valid)[0]
        if local_ids.numel() == 0:
            continue
        view_mask = rays.ray_mask[local_ids]
        origins = rays.origins[local_ids][view_mask]
        directions = rays.directions[local_ids][view_mask]
        hit_faces = ray_cast(origins, directions, posed_verts, scene_faces, device)
        object_ids = torch.full((hit_faces.shape[0],), -1, dtype=torch.long, device=hit_faces.device)
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
                anchor[point_index] = rays.xyz[local_index]
        for obj, items in pending.items():
            xyz = torch.stack([item[1] for item in items])
            point_indices = torch.tensor([item[0] for item in items], dtype=torch.long, device=world_verts.device)
            obj_faces = scene_faces[face_owner == obj]
            ok, face, uv = closest_point(xyz, posed_verts, obj_faces, device)
            if bool(ok.any()):
                chosen = point_indices[ok]
                triangles[chosen] = obj_faces[face[ok].long()].long()
                weights = uv[ok]
                bary[chosen] = torch.cat((weights, 1 - weights.sum(dim=1, keepdim=True)), dim=1)
            missed = ~ok
            if bool(missed.any()):
                anchor[point_indices[missed]] = xyz[missed]
    worlds = anchor.unsqueeze(0).expand(n_frames, -1, -1).clone()
    bound = triangles[:, 0] >= 0
    if bool(bound.any()):
        corners = world_verts[:, triangles[bound]]
        worlds[:, bound] = (corners * bary[bound].view(1, -1, 3, 1)).sum(dim=2)
    return worlds, valid
