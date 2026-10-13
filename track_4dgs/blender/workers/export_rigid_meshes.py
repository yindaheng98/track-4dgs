"""Blender worker that exports rigid meshes and their poses.

The blend file is already open when this script starts.
Each mesh is rigid in object space, so triangles are written once and ``matrix_world`` is written per frame.
``--frames`` lists absolute Blender frames and is exported in that order.
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


def pose_matrices(scene, meshes, frames):
    """Write each mesh ``matrix_world`` for every Blender frame in ``frames``.

    ``matrix_world`` is read after the depsgraph evaluates that frame.
    """
    matrices = []
    n_frames = len(frames)
    for frame_index, frame in enumerate(frames):
        scene.frame_set(int(frame))
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
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scene = bpy.context.scene
    frames = list(args.frames)
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    verts, faces = export_meshes(meshes)
    matrices = pose_matrices(scene, meshes, frames)
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
