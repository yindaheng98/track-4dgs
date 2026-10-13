"""Socket protocol for one Blender still.

``render_server.py`` imports this file by path and does not import the training package.
A request is one :class:`RenderRequest`.
A reply is one :class:`RenderReply`.
``encode`` writes a message and ``decode`` reads one.
A failed render is :meth:`RenderReply.encode_error`.
The client sends ``None`` to stop the server.
Array fields travel as raw bytes so the two NumPy builds do not share a pickle layout.
The dtype and shape travel with each array.
``frame`` is the absolute Blender frame, so the server calls ``scene.frame_set``.
``focal`` is pixels. The server maps it to a lens on a 36 mm horizontal sensor.
``c2w`` is Blender camera-to-world, so the server assigns it to ``matrix_world``.
"""

from dataclasses import dataclass

import numpy as np


AUTH = b"track-4dgs-render"


@dataclass(frozen=True)
class PackedArray:
    """One NumPy array as raw bytes.

    ``dtype`` is the NumPy dtype string.
    ``shape`` is the array shape.
    ``data`` is the contiguous buffer.
    The socket carries a dict of these three fields, not this class, because the two interpreters load this file under different module names.
    """

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def pack(cls, array: np.ndarray) -> "PackedArray":
        """Copy ``array`` into one packed value. The dtype is the array's own dtype."""

        array = np.ascontiguousarray(array)
        return cls(array.dtype.str, tuple(int(n) for n in array.shape), array.tobytes())

    def unpack(self) -> np.ndarray:
        """Inverse of :meth:`pack`. The result owns its buffer."""

        array = np.frombuffer(self.data, dtype=np.dtype(self.dtype))
        return array.reshape(self.shape).copy()


def encode(obj) -> dict:
    """Pack a message. Each array becomes a :class:`PackedArray` dict."""

    encoded = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, np.ndarray):
            encoded[key] = PackedArray.pack(value).__dict__
        else:
            encoded[key] = value
    return encoded


def decode(cls, payload: dict):
    """Inverse of :func:`encode`. Array fields are those annotated as ``np.ndarray``."""

    values = {}
    for key, field in cls.__dataclass_fields__.items():
        value = payload[key]
        if field.type is np.ndarray:
            values[key] = PackedArray(**value).unpack()
        else:
            values[key] = value
    return cls(**values)


@dataclass(frozen=True)
class RenderRequest:
    """One still sent to ``render_server.py``.

    ``blend`` is the configured ``.blend`` path.
    ``frame`` is the absolute Blender frame. The server calls ``scene.frame_set``.
    ``width`` and ``height`` are the still size in pixels.
    ``focal`` is the pinhole focal length in pixels. The principal point is the image center.
    ``c2w`` is ``(4, 4)`` Blender camera-to-world.
    Blender cameras look along local ``-Z`` and local ``+Y`` is up.
    A camera that looks along ``+Z`` with image ``+Y`` downward is ``opencv_c2w @ diag(1, -1, -1, 1)``.
    """

    blend: str
    frame: int
    width: int
    height: int
    focal: float
    c2w: np.ndarray

    def encode(self) -> dict:
        """Pack this request."""

        return encode(self)

    @classmethod
    def decode(cls, payload: dict) -> "RenderRequest":
        """Inverse of :meth:`encode`."""

        return decode(cls, payload)


@dataclass(frozen=True)
class RenderReply:
    """One still for one :class:`RenderRequest`.

    ``rgb`` is ``(H, W, 3)`` linear color.
    ``depth`` and ``object_id`` are ``(H, W)``.
    Row 0 is the top of the image.
    ``depth`` is the view-layer Z pass.
    ``object_id`` is the cryptomatte float of the object name.
    """

    rgb: np.ndarray
    depth: np.ndarray
    object_id: np.ndarray

    def encode(self) -> dict:
        """Pack this reply as a successful render."""

        payload = encode(self)
        payload["ok"] = True
        return payload

    @classmethod
    def encode_error(cls, text: str) -> dict:
        """Pack a failed render."""

        return {"ok": False, "error": text}

    @classmethod
    def decode(cls, payload: dict) -> "RenderReply":
        """Inverse of :meth:`encode`. A failed render raises ``RuntimeError``."""

        if not payload.get("ok"):
            raise RuntimeError(payload.get("error", "Blender render failed"))
        return decode(cls, payload)
