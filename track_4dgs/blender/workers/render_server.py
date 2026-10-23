"""Render one :class:`render_server_protocol.RenderRequest` per socket message.

The blend file is already configured.
It is opened only when its path changes.
This file imports only the standard library, NumPy, Blender, and ``render_server_protocol.py``.
``--background`` discards the framebuffer.
RGB is read back from a PNG written with the scene view transform.
Depth and object id are read back from Raw EXRs.
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
MIN_SAMPLES = 64


def configure_render(scene) -> None:
    """Render with Cycles, and keep a photographic view transform on the RGB pass.

    The sample count is raised to 64 when the blend file is lower.
    A higher count is left as authored.
    Adaptive sampling is off so that count is actually used.
    Denoising stays on for the beauty pass.
    ``Raw`` and ``Standard`` become AgX, or Filmic when AgX is not in this Blender.
    Any other view transform, its look, and the exposure stay with the blend file.
    """

    scene.render.engine = "CYCLES"
    cycles = scene.cycles
    if cycles.samples < MIN_SAMPLES:
        cycles.samples = MIN_SAMPLES
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = True
    denoiser = {item.identifier for item in cycles.bl_rna.properties["denoiser"].enum_items}
    if "OPENIMAGEDENOISE" in denoiser:
        cycles.denoiser = "OPENIMAGEDENOISE"
    view_layer = scene.view_layers[0]
    if 0 < view_layer.samples < MIN_SAMPLES:
        view_layer.samples = 0
    view_layer.cycles.use_denoising = True
    view = scene.view_settings
    if view.view_transform not in {"Raw", "Standard"}:
        return
    names = {item.identifier for item in view.bl_rna.properties["view_transform"].enum_items}
    for name in ("AgX", "Filmic"):
        if name in names:
            view.view_transform = name
            view.look = "None"
            return


def check_textures() -> None:
    """Raise when a linked image did not load.

    A missing file renders as magenta and is not a usable still.
    """

    missing = sorted({img.filepath for img in bpy.data.images if img.filepath and tuple(img.size) == (0, 0)})
    if missing:
        raise RuntimeError(f"missing textures: {missing}")


def add_file_output(tree, layers, scratch: Path, socket: str, source: str, stem: str, *, png: bool) -> None:
    """Write one compositor input into ``scratch``.

    ``png`` follows the scene view transform, look, and exposure.
    Data passes stay Raw so depth and cryptomatte are not graded.
    """

    node = tree.nodes.new("CompositorNodeOutputFile")
    node.directory = str(scratch) + os.sep
    node.file_name = ""
    node.format.media_type = "IMAGE"
    if png:
        node.format.file_format = "PNG"
        node.format.color_mode = "RGBA"
        node.format.color_depth = "16"
        node.format.color_management = "FOLLOW_SCENE"
    else:
        node.format.file_format = "OPEN_EXR"
        node.format.color_mode = "BW" if socket == "FLOAT" else "RGBA"
        node.format.color_depth = "32"
        node.format.color_management = "OVERRIDE"
        node.format.view_settings.view_transform = "Raw"
    item = node.file_output_items.new(socket, stem)
    tree.links.new(layers.outputs[source], node.inputs[item.name])


def setup_blend(path: str, scratch: Path) -> None:
    """Open ``path`` and write rgb, depth, and object id into ``scratch``.

    The blend file keeps its objects, materials, and world.
    Cycles replaces the render engine.
    The compositor is replaced so the three passes can be read back.
    """

    bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
    check_textures()
    scene = bpy.context.scene
    configure_render(scene)
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
    add_file_output(tree, layers, scratch, "RGBA", "Image", "rgb", png=True)
    add_file_output(tree, layers, scratch, "FLOAT", "Depth", "depth", png=False)
    add_file_output(tree, layers, scratch, "RGBA", "CryptoObject00", "id", png=False)


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


def read_image(path: Path, colorspace: str | None = None) -> np.ndarray:
    """Read one compositor image.

    Blender stores the first pixel row at the bottom of the image.
    The returned array has row 0 at the top.
    ``colorspace`` replaces the file's role before the pixels are read.
    ``Non-Color`` keeps a display-referred PNG as stored, in ``[0, 1]``.
    """

    image = bpy.data.images.load(str(path), check_existing=False)
    if colorspace is not None:
        image.colorspace_settings.name = colorspace
    width, height = image.size
    buffer = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    bpy.data.images.remove(image)
    return np.flipud(buffer.reshape(height, width, -1))


def render_still(scratch: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the current scene. Returns rgb, depth, and object id."""

    rgb_png = scratch / "rgb.png"
    depth_exr = scratch / "depth.exr"
    id_exr = scratch / "id.exr"
    for path in (rgb_png, depth_exr, id_exr):
        path.unlink(missing_ok=True)
    bpy.context.view_layer.update()
    bpy.ops.render.render(write_still=False)
    for path in (rgb_png, depth_exr, id_exr):
        if not path.is_file():
            raise RuntimeError(f"render did not write {path}")
    rgb = read_image(rgb_png, "Non-Color")[..., :3]
    depth = read_image(depth_exr)[..., 0]
    object_id = read_image(id_exr)[..., 0]
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
