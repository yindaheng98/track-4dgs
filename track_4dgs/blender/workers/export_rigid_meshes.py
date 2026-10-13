"""Blender worker that exports rigid meshes and their poses.

The blend file is already open when this script starts.
Each mesh is rigid in object space, so triangles are written once and ``matrix_world`` is written per frame.
Dataset frame ``i`` is Blender frame ``scene.frame_start + (start_number - 1) + i * frame_step``.
``--start-number`` and ``--frame-step`` match ``nvs2colmap.blender``.
Every such frame through ``scene.frame_end`` is exported.
The export is one ``pickle.dumps`` byte string on the connection, then this process exits.
"""

import argparse
import pickle
import sys
from multiprocessing.connection import Client

import bpy

AUTH = b"track-4dgs-export"


def export_meshes(meshes):
    """Export object-local vertices and loop triangles for each mesh."""
    verts = []
    faces = []
    for obj in meshes:
        mesh = obj.data
        mesh.calc_loop_triangles()
        verts.append([[vertex.co.x, vertex.co.y, vertex.co.z] for vertex in mesh.vertices])
        faces.append([list(triangle.vertices) for triangle in mesh.loop_triangles])
    return verts, faces


def pose_matrices(scene, meshes, n_frames, frame_base, frame_step):
    """Write each mesh ``matrix_world`` for every dataset frame.

    ``matrix_world`` is read after the depsgraph evaluates that frame.
    """
    matrices = []
    for frame_index in range(n_frames):
        scene.frame_set(frame_base + int(frame_index) * frame_step)
        bpy.context.evaluated_depsgraph_get()
        frame_matrices = []
        for obj in meshes:
            world = obj.matrix_world
            frame_matrices.append([[world[row][col] for col in range(4)] for row in range(4)])
        matrices.append(frame_matrices)
        if frame_index % 25 == 0:
            print(f"posed frame {frame_index + 1}/{n_frames}", flush=True)
    return matrices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--start-number", type=int, default=1)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scene = bpy.context.scene
    frame_base = scene.frame_start + args.start_number - 1
    if frame_base > scene.frame_end:
        n_frames = 0
    else:
        n_frames = (scene.frame_end - frame_base) // args.frame_step + 1
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    verts, faces = export_meshes(meshes)
    matrices = pose_matrices(scene, meshes, n_frames, frame_base, args.frame_step)
    payload = pickle.dumps({
        "verts": verts,
        "faces": faces,
        "matrices": matrices,
    }, protocol=pickle.HIGHEST_PROTOCOL)
    conn = Client(("127.0.0.1", args.port), authkey=AUTH)
    try:
        conn.send(payload)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
