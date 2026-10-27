"""Blender worker that exports evaluated meshes one frame at a time.

The blend file is already open when this script starts.
The first socket message is one :class:`export_meshes_protocol.MeshTopology`.
Each following message is one :class:`export_meshes_protocol.MeshFrame`, in frame order.
Positions include deformation and the object transform.
A frame whose vertex count or triangle indices differ from the first frame aborts the export.
Frames are ``range(frame_start, frame_end + 1, frame_step)``.
``frame_start`` defaults to 1.
``frame_end`` and ``frame_step`` default to the open scene.
This process exits after the last frame.
"""

import argparse
import sys
from multiprocessing.connection import Client
from pathlib import Path

import bpy

protocol = sys.path.insert(0, str(Path(__file__).resolve().parent)) or __import__("export_meshes_protocol")

AUTH = b"track-4dgs-export"


def evaluated_world(obj, depsgraph):
    """World positions and loop triangles of ``obj`` after ``depsgraph`` evaluates it."""

    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        matrix = eval_obj.matrix_world
        coords = []
        for vertex in mesh.vertices:
            co = matrix @ vertex.co
            coords.append([co.x, co.y, co.z])
        return coords, [list(triangle.vertices) for triangle in mesh.loop_triangles]
    finally:
        eval_obj.to_mesh_clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--frame_start", type=int, default=1)
    parser.add_argument("--frame_end", type=int)
    parser.add_argument("--frame_step", type=int)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scene = bpy.context.scene
    scene.frame_start = args.frame_start
    if args.frame_end is not None:
        scene.frame_end = args.frame_end
    if args.frame_step is not None:
        scene.frame_step = args.frame_step
    frames = list(range(scene.frame_start, scene.frame_end + 1, scene.frame_step))
    if not frames:
        raise SystemExit(
            f"no Blender frames from {scene.frame_start} to {scene.frame_end} step {scene.frame_step}"
        )
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    object_id = []
    item_ids = {}
    for obj in meshes:
        root = obj
        while root.parent is not None:
            root = root.parent
        if root.name not in item_ids:
            item_ids[root.name] = len(item_ids)
        object_id.append(item_ids[root.name])
    conn = Client(("127.0.0.1", args.port), authkey=AUTH)
    try:
        reference = None
        for frame in frames:
            scene.frame_set(int(frame))
            depsgraph = bpy.context.evaluated_depsgraph_get()
            verts = []
            faces = []
            for obj in meshes:
                coords, triangles = evaluated_world(obj, depsgraph)
                verts.append(coords)
                faces.append(triangles)
            if reference is None:
                reference = ([len(coords) for coords in verts], faces)
                conn.send(protocol.MeshTopology(
                    faces=faces,
                    object_id=object_id,
                    frame_start=scene.frame_start,
                    frame_end=scene.frame_end,
                    frame_step=scene.frame_step,
                ).encode())
            else:
                counts, ref_faces = reference
                for obj, coords, triangles, count, ref in zip(meshes, verts, faces, counts, ref_faces):
                    if len(coords) != count or triangles != ref:
                        raise RuntimeError(
                            f"{obj.name} topology changed at frame {int(frame)}: "
                            f"{count} vertices, {len(ref)} triangles became "
                            f"{len(coords)} vertices, {len(triangles)} triangles"
                        )
            conn.send(protocol.MeshFrame(verts=verts).encode())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
