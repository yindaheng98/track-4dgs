"""Track query pixels by binding them to evaluated meshes in a blend file."""

import subprocess
from collections.abc import Iterator, Sequence
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Optional

import torch
from gaussian_splatting.dataset import CameraDataset

from ..tracker import AbstractBatchPointTracker, Query, Track
from .kernels import project_visibility, track_points
from .workers import MeshFrame, MeshTopology

AUTH = b"track-4dgs-export"


def evaluated_meshes(
        blend: str,
        blender: str = "blender",
        frame_start: int = 1,
        frame_end: Optional[int] = None,
        frame_step: Optional[int] = None) -> tuple[MeshTopology, Iterator[MeshFrame]]:
    """Export evaluated meshes from a blend file.

    Returns the topology and an iterator of one :class:`MeshFrame` per later frame.
    ``MeshTopology.verts`` is the first frame.
    The iterator can be consumed only once.
    Frames are ``range(frame_start, frame_end + 1, frame_step)``.
    ``frame_start`` defaults to 1.
    ``frame_end`` and ``frame_step`` default to the blend scene.
    """

    script = Path(__file__).parent / "workers" / "export_meshes.py"
    listener = Listener(("127.0.0.1", 0), authkey=AUTH)
    server = None
    connection = None
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
        topology = MeshTopology.decode(connection.recv())
    except BaseException:
        if connection is not None:
            connection.close()
        if server is not None and server.poll() is None:
            server.kill()
            server.wait()
        listener.close()
        raise
    n_frames = len(range(topology.frame_start, topology.frame_end + 1, topology.frame_step)) - 1

    def iter_frames(
            connection: Connection,
            n_frames: int,
            server: subprocess.Popen,
            listener: Listener,
            command: list) -> Iterator[MeshFrame]:
        """Read ``n_frames`` mesh frames, then close the Blender export.

        ``connection`` receives one :class:`MeshFrame` per frame.
        ``server`` is the Blender process, and ``command`` is the command that started it.
        ``listener`` is closed when the frames have been read or the read fails.
        """

        try:
            try:
                for _ in range(n_frames):
                    yield MeshFrame.decode(connection.recv())
            finally:
                connection.close()
        except BaseException:
            if server.poll() is None:
                server.kill()
                server.wait()
            raise
        finally:
            listener.close()
        returncode = server.wait()
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, command)

    return topology, iter_frames(connection, n_frames, server, listener, command)


class BlenderPointTracker(AbstractBatchPointTracker):
    """Track query pixels by binding them to evaluated meshes in a blend file.

    The export returns triangles and the first frame together, then one later frame at a time.
    Rays, object votes, and surface snaps run against that frame's mesh.
    A snap stores barycentric coordinates and replays them on later frames.
    ``Track.xyz`` is those positions as ``(F, N, 3)`` blend-world coordinates, one tensor shared by every view.
    Every view of a point shares one query frame.
    Visibility is an any-hit test from each camera to each posed point.
    Exported frames are ``range(frame_start, frame_end + 1, frame_step)``.
    The resolved bounds are stored on ``frame_start``, ``frame_end``, and ``frame_step``.
    ``verts`` is one ``(F, V_i, 3)`` world-space tensor per mesh.
    ``n_frames`` is ``F``.
    ``object_id`` is the item index of each exported mesh, parallel to ``verts`` and ``faces``.
    ``frame_start`` defaults to 1.
    ``frame_end`` and ``frame_step`` default to the blend scene.
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
        topology, frames = evaluated_meshes(blend, blender, frame_start, frame_end, frame_step)
        self.frame_start = int(topology.frame_start)
        self.frame_end = int(topology.frame_end)
        self.frame_step = int(topology.frame_step)
        self.faces = [torch.tensor(faces, dtype=torch.int32).reshape(-1, 3) for faces in topology.faces]
        self.object_id = torch.tensor(topology.object_id, dtype=torch.long)
        self.verts = []
        for coords in topology.verts:
            row = torch.tensor(coords, dtype=torch.float32) if coords else torch.zeros((0, 3), dtype=torch.float32)
            buf = row.new_empty((self.n_frames, row.shape[0], 3))
            buf[0] = row
            self.verts.append(buf)
        for frame_index, frame in enumerate(frames, start=1):
            for mesh, coords in zip(self.verts, frame.verts):
                if coords:
                    mesh[frame_index] = torch.tensor(coords, dtype=mesh.dtype)

    @property
    def n_frames(self) -> int:
        """Number of exported frames."""

        return len(range(self.frame_start, self.frame_end + 1, self.frame_step))

    def to(self, device) -> 'BlenderPointTracker':
        """Move the exported meshes onto ``device`` and run Warp there."""

        self.device = torch.device(device)
        self.verts = [verts.to(self.device) for verts in self.verts]
        self.faces = [faces.to(self.device) for faces in self.faces]
        self.object_id = self.object_id.to(self.device)
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
        worlds, valid, world_verts, scene_faces, face_owner = track_points(
            self.verts, self.faces, query, frames, str(self.device),
        )
        points, visibility = project_visibility(
            query, frames, worlds, world_verts, scene_faces, valid, self.eps, str(self.device),
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
