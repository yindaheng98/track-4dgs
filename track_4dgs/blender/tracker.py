"""Track query pixels by binding them to rigid meshes in a blend file."""

import pickle
import subprocess
from collections.abc import Sequence
from multiprocessing.connection import Listener
from pathlib import Path

import torch
import warp as wp
from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractBatchPointTracker, Query, Track
from .kernels import track_points

AUTH = b"track-4dgs-export"


class BlenderPointTracker(AbstractBatchPointTracker):
    """Track query pixels by binding them to rigid meshes in a blend file.

    Blender exports triangles once, at construction, and a ``matrix_world`` per frame.
    Rays, object votes, and surface snaps run against that posed mesh.
    Every view of a point shares one query frame.
    Visibility is an any-hit test from each camera to each posed point.
    Dataset frame ``i`` is Blender frame ``scene.frame_start + (start_number - 1) + i * frame_step``.
    ``start_number`` and ``frame_step`` match ``nvs2colmap.blender --start-number`` and ``--frame-step``.
    ``start_number`` is 1-based and defaults to 1, so the first dataset frame is ``scene.frame_start``.
    Cameras use the blend world, look along +Z, and treat image +Y as downward.
    ``eps`` is the visibility depth tolerance in world units.
    """

    def __init__(
            self,
            blend: str,
            blender: str = "blender",
            eps: float = 1e-2,
            start_number: int = 1,
            frame_step: int = 1):
        self.eps = eps
        script = Path(__file__).parent / "workers" / "export_rigid_meshes.py"
        listener = Listener(("127.0.0.1", 0), authkey=AUTH)
        server = None
        try:
            command = [
                blender,
                str(Path(blend).absolute()),
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(script),
                "--",
                "--port",
                str(listener.address[1]),
                "--start-number",
                str(start_number),
                "--frame-step",
                str(frame_step),
            ]
            server = subprocess.Popen(command)
            connection = listener.accept()
            try:
                payload = connection.recv()
            finally:
                connection.close()
        except BaseException:
            if server is not None and server.poll() is None:
                server.kill()
                server.wait()
            raise
        finally:
            listener.close()
        returncode = server.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)
        result = pickle.loads(payload)
        self.verts = [torch.tensor(obj_verts, dtype=torch.float32).reshape(-1, 3) for obj_verts in result["verts"]]
        self.faces = [torch.tensor(obj_faces, dtype=torch.int32).reshape(-1, 3) for obj_faces in result["faces"]]
        self.matrices = torch.tensor(result["matrices"], dtype=torch.float32)

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
        wp.init()
        device = "cuda" if wp.is_cuda_available() else "cpu"
        points, visibility, valid = track_points(
            self.verts, self.faces, self.matrices, query, frames, self.eps, device,
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
