"""One background Blender process that renders stills."""

import subprocess
from multiprocessing.connection import Listener
from pathlib import Path

import torch

from .workers.render_server_protocol import RenderReply, RenderRequest

AUTH = b"track-4dgs-render"  # same as in workers/render_server.py


class RenderServer:
    """One background Blender process serving :class:`RenderRequest`.

    ``__init__`` listens on ``127.0.0.1`` and starts ``workers/render_server.py``.
    Blender connects back with ``AUTH``.
    ``device`` is a :class:`torch.device`. Its ``type`` and ``index`` are passed as ``--device-type`` and ``--device-index``.
    A ``cuda`` device with no index uses :func:`torch.cuda.current_device`.
    :meth:`render` sends one request and returns the reply.
    :meth:`__del__` sends ``None`` and waits for Blender to exit.
    """

    def __init__(self, blender: str = "blender", device: str | torch.device = "cuda"):
        self.device = torch.device(device)
        script = Path(__file__).resolve().parent / "workers" / "render_server.py"
        listener = Listener(("127.0.0.1", 0), authkey=AUTH)
        command = [
            blender,
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            "--port",
            str(listener.address[1]),
            "--device-type",
            self.device.type,
        ]
        if self.device.type == "cuda":
            index = torch.cuda.current_device() if self.device.index is None else self.device.index
            command.extend(("--device-index", str(index)))
        try:
            self.server = subprocess.Popen(command)
            self.connection = listener.accept()
        except BaseException:
            server = getattr(self, "server", None)
            if server is not None and server.poll() is None:
                server.kill()
                server.wait()
            raise
        finally:
            listener.close()

    def render(self, request: RenderRequest) -> RenderReply:
        """Render one still."""

        self.connection.send(request.encode())
        return RenderReply.decode(self.connection.recv())

    def __del__(self):
        connection = getattr(self, "connection", None)
        server = getattr(self, "server", None)
        self.connection = None
        self.server = None
        try:
            if connection is not None:
                connection.send(None)
                connection.close()
        finally:
            if server is not None:
                server.wait()
