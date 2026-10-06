import json
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
from gaussian_splatting.dataset import CameraDataset

from track_4dgs.tracker import AbstractBatchPointTracker, Query, Track
from track_4dgs.utils import project_points


def ray(camera, uv):
    direction = torch.stack((
        (uv[0] + 0.5 - camera.K[0, 2]) / camera.K[0, 0],
        (uv[1] + 0.5 - camera.K[1, 2]) / camera.K[1, 1],
        uv.new_tensor(1.0),
    ))
    direction = camera.R.T @ direction
    return camera.camera_center, direction / direction.norm()


def triangulate(origins, directions):
    eye = torch.eye(3, dtype=origins[0].dtype)
    system = torch.zeros((3, 3), dtype=origins[0].dtype)
    target = torch.zeros(3, dtype=origins[0].dtype)
    for origin, direction in zip(origins, directions):
        onto_plane = eye - direction[:, None] * direction[None, :]
        system = system + onto_plane
        target = target + onto_plane @ origin
    return torch.linalg.solve(system, target)


class BlenderPointTracker(AbstractBatchPointTracker):
    """Track query pixels against a Blender scene.

    Rays and triangulation run here. Every view of a point shares one
    ``frame_indices`` value. Those simultaneous rays are triangulated, then one
    Blender process casts them, votes for an object, snaps the point onto its
    surface, and follows ``matrix_world`` while testing visibility. Dataset
    frame ``i`` is Blender frame ``scene.frame_start + i``. Camera ``R`` / ``T``
    are in the same world as the ``.blend`` file, looking along +Z with image
    +Y downward. ``eps`` is
    the visibility depth tolerance.
    """

    def __init__(self, blend: str, blender: str = "blender", eps: float = 1e-2, frame_start: int | None = None, frame_step: int = 1):
        self.blend = blend
        self.blender = blender
        self.eps = eps
        self.frame_start = frame_start
        self.frame_step = frame_step

    def _request(self, query: Query, frames: Sequence[CameraDataset]):
        n_views = query.points.shape[0]
        points = []
        for point_index in range(query.points.shape[1]):
            frame_index = int(query.frame_indices[0, point_index])
            origins, directions = [], []
            for view_index in range(n_views):
                if not bool(query.in_image[view_index, point_index]):
                    continue
                origin, direction = ray(frames[frame_index][view_index], query.points[view_index, point_index])
                origins.append(origin.detach().cpu())
                directions.append(direction.detach().cpu())
            if len(origins) < 2:
                points.append(None)
                continue
            points.append({
                "frame": frame_index,
                "xyz": triangulate(origins, directions).tolist(),
                "origins": torch.stack(origins).tolist(),
                "directions": torch.stack(directions).tolist(),
            })
        centers = [
            [dataset[view_index].camera_center.detach().cpu().tolist() for view_index in range(n_views)]
            for dataset in frames
        ]
        request = {"eps": self.eps, "frame_step": self.frame_step, "points": points, "centers": centers}
        if self.frame_start is not None:
            request["frame_start"] = self.frame_start
        return request

    @torch.no_grad()
    def track_batch(
            self,
            query: Query,
            frames: Sequence[CameraDataset]) -> Sequence[Track]:
        assert torch.all(query.frame_indices == query.frame_indices[0])
        n_views, n_points, _ = query.points.shape
        n_frames = len(frames)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            request_path = folder / "request.json"
            result_path = folder / "result.json"
            request_path.write_text(json.dumps(self._request(query, frames), separators=(",", ":")))
            subprocess.run(
                [
                    self.blender,
                    str(Path(self.blend).absolute()),
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("track.py")),
                    "--",
                    "--input",
                    str(request_path),
                    "--output",
                    str(result_path),
                ],
                check=True,
            )
            result = json.loads(result_path.read_text())

        worlds = torch.tensor(result["worlds"], dtype=query.points.dtype)
        hit = torch.tensor(result["visibility"], dtype=query.points.dtype)
        points = query.points.new_zeros((n_views, n_frames, n_points, 2))
        visibility = query.points.new_zeros((n_views, n_frames, n_points))
        confidence = query.points.new_tensor(result["confidence"]).expand(n_views, n_frames, n_points).clone()
        mask = torch.tensor(result["mask"], dtype=torch.bool, device=query.points.device).expand(n_frames, n_points)
        for frame_index, dataset in enumerate(frames):
            for view_index in range(n_views):
                camera = dataset[view_index]
                uv, seen = project_points(worlds[frame_index].to(device=camera.R.device, dtype=camera.R.dtype), camera)
                points[view_index, frame_index] = uv.to(points.device)
                visibility[view_index, frame_index] = (hit[frame_index, view_index].to(seen.device) * seen.to(hit.dtype)).to(visibility.device)

        return [
            Track(
                points=points[view_index],
                visibility=visibility[view_index],
                confidence=confidence[view_index],
                mask=mask,
            )
            for view_index in range(n_views)
        ]
