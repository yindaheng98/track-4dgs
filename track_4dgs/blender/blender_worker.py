"""Blender worker that binds triangulated rays and exports rigid poses.

The blend file is already open when this script starts.
Visibility is not cast here.
Each mesh is rigid in object space, so triangles are written once and ``matrix_world`` is written per frame.
Dataset frame ``i`` is Blender frame ``frame_start + i * frame_step``.
A missing ``frame_start`` uses ``scene.frame_start``.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import bpy
from mathutils import Vector


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


def bind_points(scene, request, meshes, frame_base, frame_step):
    """Snap each triangulated point onto a mesh, or leave it in world space.

    Each ray votes for the first object it hits.
    A miss votes for empty space.
    The winner must be an exported mesh, and it must beat both empty space and the runner-up.
    A successful snap stores object-local coordinates, with confidence equal to the winning vote fraction.
    Otherwise the binding stays at the triangulated world position with object index ``-1``.
    Confidence is then the fraction of rays that did not vote for the rejected winner.
    Points without a ray bundle stay at ``None`` with mask false.
    A binding with object index ``-1`` is still masked on.
    """
    index_of = {obj.name: index for index, obj in enumerate(meshes)}
    n_points = len(request["points"])
    bindings = [None] * n_points
    confidence = [0.0] * n_points
    mask = [False] * n_points
    grouped = defaultdict(list)
    for index, point in enumerate(request["points"]):
        if point is not None:
            grouped[point["frame"]].append(index)
    for frame, indices in grouped.items():
        scene.frame_set(frame_base + int(frame) * frame_step)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for index in indices:
            point = request["points"][index]
            xyz = Vector(point["xyz"])
            votes = Counter()
            misses = 0
            for origin, direction in zip(point["origins"], point["directions"]):
                hit, _, _, _, obj, _ = scene.ray_cast(depsgraph, Vector(origin), Vector(direction))
                if hit:
                    votes[obj.name] += 1
                else:
                    misses += 1
            ranking = votes.most_common(2)
            if ranking:
                name, count = ranking[0]
            else:
                name, count = None, 0
            second = ranking[1][1] if len(ranking) > 1 else 0
            n_rays = len(point["origins"])
            snapped = None
            if name in index_of and count > misses and count > second:
                obj = bpy.data.objects[name].evaluated_get(depsgraph)
                ok, location, _, _ = obj.closest_point_on_mesh(obj.matrix_world.inverted() @ xyz)
                if ok:
                    snapped = [index_of[name], location.x, location.y, location.z]
                    confidence[index] = count / n_rays
            if snapped is None:
                snapped = [-1, *point["xyz"]]
                confidence[index] = (n_rays - count) / n_rays
            bindings[index] = snapped
            mask[index] = True
    return bindings, confidence, mask


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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    request = json.loads(args.input.read_text())
    scene = bpy.context.scene
    if request.get("frame_start") is None:
        frame_base = scene.frame_start
    else:
        frame_base = int(request["frame_start"])
    frame_step = int(request.get("frame_step", 1))
    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    verts, faces = export_meshes(meshes)
    bindings, confidence, mask = bind_points(scene, request, meshes, frame_base, frame_step)
    matrices = pose_matrices(scene, meshes, request["n_frames"], frame_base, frame_step)
    args.output.write_text(json.dumps({
        "verts": verts,
        "faces": faces,
        "matrices": matrices,
        "bindings": bindings,
        "confidence": confidence,
        "mask": mask,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
