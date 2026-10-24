"""Track query pixels by binding them to rigid meshes in a blend file."""

import pickle
import subprocess
from collections.abc import Sequence
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractBatchPointTracker, Query, Track
from .kernels import track_points
from .workers.export_rigid_meshes_protocol import RigidMeshes

AUTH = b"track-4dgs-export"


def export_rigid_meshes(
        blend: str,
        blender: str = "blender",
        frame_start: int = 1,
        frame_end: Optional[int] = None,
        frame_step: Optional[int] = None) -> tuple:
    """Export rigid meshes and one ``matrix_world`` per frame from a blend file.

    Frames are ``range(frame_start, frame_end + 1, frame_step)``.
    ``frame_start`` defaults to 1. ``frame_end`` and ``frame_step`` default to the blend scene.
    Returns one :class:`RigidMeshes`, per-frame ``matrices``, and the resolved ``frame_start``, ``frame_end``, and ``frame_step``.
    """

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
            "--frame_start",
            str(frame_start),
            *(["--frame_end", str(frame_end)] if frame_end is not None else []),
            *(["--frame_step", str(frame_step)] if frame_step is not None else []),
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
    return (
        RigidMeshes.decode(result),
        result["matrices"],
        int(result["frame_start"]),
        int(result["frame_end"]),
        int(result["frame_step"]),
    )


class BlenderPointTracker(AbstractBatchPointTracker):
    """Track query pixels by binding them to rigid meshes in a blend file.

    Blender exports triangles once, at construction, and a ``matrix_world`` per frame.
    Rays, object votes, and surface snaps run against that posed mesh.
    Every view of a point shares one query frame.
    Visibility is an any-hit test from each camera to each posed point.
    Exported frames are ``range(frame_start, frame_end + 1, frame_step)``.
    The resolved bounds are stored on ``frame_start``, ``frame_end``, and ``frame_step``.
    ``object_id`` is the item index of each exported mesh, parallel to ``verts`` and ``faces``.
    ``frame_start`` defaults to 1. ``frame_end`` and ``frame_step`` default to the blend scene.
    Cameras use the blend world, look along +Z, and treat image +Y as downward.
    ``eps`` is the visibility depth tolerance in world units.
    """

    def __init__(
            self,
            blend: str,
            blender: str = "blender",
            eps: float = 1e-2,
            frame_start: int = 1,
            frame_end: Optional[int] = None,
            frame_step: Optional[int] = None):
        self.eps = eps
        self.device = torch.device("cpu")
        self.load_frames(blend, blender, frame_start, frame_end, frame_step)

    def load_frames(
            self,
            blend: str,
            blender: str = "blender",
            frame_start: int = 1,
            frame_end: Optional[int] = None,
            frame_step: Optional[int] = None) -> None:
        meshes, matrices, self.frame_start, self.frame_end, self.frame_step = export_rigid_meshes(
            blend, blender, frame_start, frame_end, frame_step)
        self.verts = [torch.tensor(obj_verts, dtype=torch.float32).reshape(-1, 3) for obj_verts in meshes.verts]
        self.faces = [torch.tensor(obj_faces, dtype=torch.int32).reshape(-1, 3) for obj_faces in meshes.faces]
        self.object_id = torch.tensor(meshes.object_id, dtype=torch.long)
        self.matrices = torch.tensor(matrices, dtype=torch.float32)

    def to(self, device) -> 'BlenderPointTracker':
        """Move the exported meshes onto ``device`` and run Warp there."""

        self.device = torch.device(device)
        self.verts = [verts.to(self.device) for verts in self.verts]
        self.faces = [faces.to(self.device) for faces in self.faces]
        self.object_id = self.object_id.to(self.device)
        self.matrices = self.matrices.to(self.device)
        return self

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
        points, visibility, valid = track_points(
            self.verts, self.faces, self.matrices, query, frames, self.eps, str(self.device),
        )
        confidence = query.points.new_ones((n_views, n_frames, n_points))
        mask = valid.to(dtype=torch.bool).expand(n_frames, n_points)
        return [
            Track(
                points=points[view_index],
                visibility=visibility[view_index],
                confidence=confidence[view_index],
                mask=mask,
            )
            for view_index in range(n_views)
        ]
