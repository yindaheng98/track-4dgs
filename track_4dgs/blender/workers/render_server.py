"""Render one :class:`render_server_protocol.RenderRequest` per socket message.

The blend file is already configured.
It is opened only when its path changes.
This file imports only the standard library, NumPy, Blender, and ``render_server_protocol.py``.
``--background`` discards the framebuffer.
Color is read back from the Filmic PNG.
Depth and object id are read back from EXR.
"""

import argparse
import os
import sys
import tempfile
import traceback
from multiprocessing.connection import Client
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

protocol = sys.path.insert(0, str(Path(__file__).resolve().parent)) or __import__("render_server_protocol")

AUTH = b"track-4dgs-render" # same as in ../render_server.py
SENSOR_WIDTH_MM = 36.0


def setup_blend(path: str, scratch: Path) -> None:
    """Open ``path`` and write rgb, depth, and object id into ``scratch``.

    The blend file keeps its objects, materials, and render engine.
    This replaces the scene compositor so the three passes can be read back.
    """

    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    scene = bpy.context.scene
    # Kubric saves the color image from a Filmic PNG and keeps the EXR linear.
    # https://github.com/google-research/kubric/blob/main/kubric/renderer/blender.py
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.render.use_compositing = True
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = True
    view_layer.use_pass_cryptomatte_object = True

    tree = bpy.data.node_groups.new("RenderComp", "CompositorNodeTree")
    scene.compositing_node_group = tree
    tree.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    layers = tree.nodes.new("CompositorNodeRLayers")
    layers.layer = view_layer.name
    group_out = tree.nodes.new("NodeGroupOutput")
    tree.links.new(layers.outputs["Image"], group_out.inputs["Image"])
    for socket, source, color_mode, color_depth, stem in (
        ("FLOAT", "Depth", "BW", "32", "depth"),
        ("RGBA", "CryptoObject00", "RGBA", "32", "id"),
    ):
        node = tree.nodes.new("CompositorNodeOutputFile")
        node.directory = str(scratch) + os.sep
        node.file_name = ""
        node.format.media_type = "IMAGE"
        node.format.file_format = "OPEN_EXR"
        node.format.color_mode = color_mode
        node.format.color_depth = color_depth
        node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
        item = node.file_output_items.new(socket, stem)
        tree.links.new(layers.outputs[source], node.inputs[item.name])


def apply_camera(request) -> None:
    """Point the scene camera at ``request`` and set the still size.

    Animation on the camera object and its lens is cleared so the blend file's camera action does not override this pose.
    """

    scene = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None or cam_obj.type != "CAMERA":
        raise RuntimeError(f"{request.blend} has no scene camera")
    cam_obj.animation_data_clear()
    cam = cam_obj.data
    cam.animation_data_clear()
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = SENSOR_WIDTH_MM
    cam.lens = float(request.focal) * cam.sensor_width / int(request.width)
    c2w = np.asarray(request.c2w, dtype=np.float64).reshape(4, 4)
    cam_obj.matrix_world = Matrix(c2w.tolist())
    scene.render.resolution_x = int(request.width)
    scene.render.resolution_y = int(request.height)
    scene.render.resolution_percentage = 100


def read_display_png(path: Path) -> np.ndarray:
    """Read one Filmic PNG as ``(H, W, 3)`` float in ``[0, 1]``.

    ``Non-Color`` keeps the file bytes.
    Blender stores the first row at the bottom of ``image.pixels``.
    """

    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    buffer = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    bpy.data.images.remove(image)
    return np.flipud(buffer.reshape(height, width, -1))[..., :3]


def read_exr(path: Path) -> np.ndarray:
    """Read one compositor EXR.

    Blender stores the first pixel row at the bottom of the image.
    The returned array has row 0 at the top.
    """

    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    buffer = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    bpy.data.images.remove(image)
    return np.flipud(buffer.reshape(height, width, -1))


def render_still(scratch: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the current scene. Returns rgb, depth, and object id."""

    depth_exr = scratch / "depth.exr"
    id_exr = scratch / "id.exr"
    png = scratch / "rgb.png"
    for path in (depth_exr, id_exr, png):
        path.unlink(missing_ok=True)
    scene = bpy.context.scene
    scene.render.filepath = str(png)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=True)
    for path in (depth_exr, id_exr, png):
        if not path.is_file():
            raise RuntimeError(f"render did not write {path}")
    rgb = read_display_png(png)
    depth = read_exr(depth_exr)[..., 0]
    object_id = read_exr(id_exr)[..., 0]
    return rgb, depth, object_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scratch = Path(tempfile.mkdtemp(prefix="track4dgs-render-"))
    conn = Client(("127.0.0.1", args.port), authkey=AUTH)
    loaded_blend = None
    try:
        while True:
            message = conn.recv()
            if message is None:
                break
            try:
                request = protocol.RenderRequest.decode(message)
                if request.blend != loaded_blend:
                    setup_blend(request.blend, scratch)
                    loaded_blend = request.blend
                bpy.context.scene.frame_set(int(request.frame))
                apply_camera(request)
                rgb, depth, object_id = render_still(scratch)
                conn.send(protocol.RenderReply(rgb=rgb, depth=depth, object_id=object_id).encode())
            except Exception:
                conn.send(protocol.RenderReply.encode_error(traceback.format_exc()))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
