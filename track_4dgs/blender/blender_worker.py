"""Blender worker that exports rigid meshes and their poses.

The blend file is already open when this script starts.
Each mesh is rigid in object space, so triangles are written once and ``matrix_world`` is written per frame.
Dataset frame ``i`` is Blender frame ``frame_start + i * frame_step``.
A missing ``--frame-start`` uses ``scene.frame_start``.
The export is written with pickle.
"""

import argparse
import pickle
import sys
from pathlib import Path

import bpy


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
    parser.add_argument("--n-frames", type=int, required=True)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    scene = bpy.context.scene
    frame_base = scene.frame_start if args.frame_start is None else args.frame_start
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    verts, faces = export_meshes(meshes)
    matrices = pose_matrices(scene, meshes, args.n_frames, frame_base, args.frame_step)
    args.output.write_bytes(pickle.dumps({
        "verts": verts,
        "faces": faces,
        "matrices": matrices,
    }, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    main()
