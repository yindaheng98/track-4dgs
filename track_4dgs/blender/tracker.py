"""Track query pixels by binding them to rigid meshes in a blend file."""

import pickle
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

import torch
import warp as wp
from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractBatchPointTracker, Query, Track
from .kernels import track_points


class BlenderPointTracker(AbstractBatchPointTracker):
    """Track query pixels by binding them to rigid meshes in a blend file.

    Blender exports triangles once and a ``matrix_world`` per frame.
    Rays, object votes, and surface snaps run against that posed mesh.
    Every view of a point shares one query frame.
    Visibility is an any-hit test from each camera to each posed point.
    Dataset frame ``i`` is Blender frame ``frame_start + i * frame_step``.
    ``frame_start`` left as ``None`` uses ``scene.frame_start`` from the blend file.
    Cameras use the blend world, look along +Z, and treat image +Y as downward.
    ``eps`` is the visibility depth tolerance in world units.
    """

    def __init__(
            self,
            blend: str,
            blender: str = "blender",
            eps: float = 1e-2,
            frame_start: int | None = None,
            frame_step: int = 1):
        self.blend = blend
        self.blender = blender
        self.eps = eps
        self.frame_start = frame_start
        self.frame_step = frame_step

    @torch.no_grad()
    def track_batch(
            self,
            query: Query,
            frames: Sequence[CameraDataset]) -> Sequence[Track]:
        assert torch.all(query.frame_indices == query.frame_indices[0]), (
            "every view of a point must share one frame"
        )
        n_views, n_points, _ = query.points.shape
        n_frames = len(frames)
        script = Path(__file__).with_name("blender_worker.py")
        with tempfile.TemporaryDirectory() as folder:
            result_path = Path(folder) / "result.pkl"
            command = [
                self.blender,
                str(Path(self.blend).absolute()),
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(script),
                "--",
                "--n-frames",
                str(n_frames),
                "--frame-step",
                str(self.frame_step),
                "--output",
                str(result_path),
            ]
            if self.frame_start is not None:
                command.extend(["--frame-start", str(self.frame_start)])
            subprocess.run(command, check=True)
            result = pickle.loads(result_path.read_bytes())
        verts = [torch.tensor(obj_verts, dtype=torch.float32).reshape(-1, 3) for obj_verts in result["verts"]]
        faces = [torch.tensor(obj_faces, dtype=torch.int32).reshape(-1, 3) for obj_faces in result["faces"]]
        matrices = torch.tensor(result["matrices"], dtype=torch.float32)
        wp.init()
        device = "cuda" if wp.is_cuda_available() else "cpu"
        points, visibility, valid = track_points(
            verts, faces, matrices, query, frames, self.eps, device,
        )
        confidence = query.points.new_ones((n_views, n_frames, n_points))
        mask = valid.to(device=query.points.device, dtype=torch.bool).expand(n_frames, n_points)
        return [
            Track(
                points=points[view_index],
                visibility=visibility[view_index],
                confidence=confidence[view_index],
                mask=mask,
            )
            for view_index in range(n_views)
        ]
