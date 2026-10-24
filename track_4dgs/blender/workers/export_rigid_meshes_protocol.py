"""Payload for one rigid-mesh export.

``export_rigid_meshes.py`` imports this file by path and does not import the training package.
The mesh record is one :class:`RigidMeshes`.
``encode`` writes that record and ``decode`` reads it back.
The socket carries the dict, not this class, because the two interpreters load this file under different module names.
Add a field here when the export grows. Both sides pick it up from the dataclass.
"""

from dataclasses import dataclass


def encode(obj) -> dict:
    """Pack a record as a dict of its dataclass fields."""

    return dict(obj.__dict__)


def decode(cls, payload: dict):
    """Inverse of :func:`encode`. Keys outside the dataclass are ignored."""

    return cls(**{key: payload[key] for key in cls.__dataclass_fields__})


@dataclass(frozen=True)
class RigidMeshes:
    """Object-local triangles and the item that owns each mesh.

    ``verts[i]`` is a list of ``[x, y, z]`` in that mesh's object space.
    ``faces[i]`` is a list of triangle vertex indices into ``verts[i]``.
    ``object_id[i]`` is the item that owns mesh ``i``.
    Meshes that share a root parent are one item and share one id.
    A mesh with no parent is its own item.
    Ids are dense and follow the first exported mesh of each item.
    """

    verts: list
    faces: list
    object_id: list

    def encode(self) -> dict:
        """Pack this record."""

        return encode(self)

    @classmethod
    def decode(cls, payload: dict) -> "RigidMeshes":
        """Inverse of :meth:`encode`."""

        return decode(cls, payload)
