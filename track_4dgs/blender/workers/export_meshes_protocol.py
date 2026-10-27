"""Socket messages for one evaluated-mesh export.

``export_meshes.py`` imports this file by path and does not import the training package.
The first message is one :class:`MeshTopology`, including the first frame's vertices.
Each following message is one :class:`MeshFrame` for a later frame.
``encode`` writes a message and ``decode`` reads one.
The socket carries the dict, not the class, because the two interpreters load this file under different module names.
"""

from dataclasses import dataclass


def encode(obj) -> dict:
    """Pack a message as a dict of its dataclass fields."""

    return dict(obj.__dict__)


def decode(cls, payload: dict):
    """Inverse of :func:`encode`."""

    return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class MeshTopology:
    """Triangles of every exported mesh, and the frame range those meshes share.

    ``faces[i]`` is a list of triangle vertex indices into mesh ``i``.
    ``verts`` is the first exported frame, in the same layout as :class:`MeshFrame` ``verts``.
    ``object_id[i]`` is the item that owns mesh ``i``.
    Meshes that share a root parent are one item and share one id.
    A mesh with no parent is its own item.
    Ids are dense and follow the first exported mesh of each item.
    ``frame_start``, ``frame_end``, and ``frame_step`` are the resolved Blender range.
    Exported frames are ``range(frame_start, frame_end + 1, frame_step)``.
    """

    faces: list
    verts: list
    object_id: list
    frame_start: int
    frame_end: int
    frame_step: int

    def encode(self) -> dict:
        """Pack this message."""

        return encode(self)

    @classmethod
    def decode(cls, payload: dict) -> "MeshTopology":
        """Inverse of :meth:`encode`."""

        return decode(cls, payload)


@dataclass(frozen=True)
class MeshFrame:
    """World positions of every mesh on one exported frame.

    ``verts[i]`` is a list of ``[x, y, z]`` for mesh ``i``, after deformation and the object transform.
    Vertex order matches :class:`MeshTopology` ``faces[i]``.
    """

    verts: list

    def encode(self) -> dict:
        """Pack this message."""

        return encode(self)

    @classmethod
    def decode(cls, payload: dict) -> "MeshFrame":
        """Inverse of :meth:`encode`."""

        return decode(cls, payload)
