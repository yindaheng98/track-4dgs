"""Bind triangulated rays to scene objects and sample their motion.

Executed by Blender, not by the project interpreter. The blend file is already
open: ``blender scene.blend --background --python track.py -- --input ... --output ...``.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    request = json.loads(args.input.read_text())

    scene = bpy.context.scene
    eps = request["eps"]

    frame_base = scene.frame_start if request.get("frame_start") is None else int(request["frame_start"])
    frame_step = int(request.get("frame_step", 1))

    def use_frame(frame_index):
        scene.frame_set(frame_base + int(frame_index) * frame_step)
        return bpy.context.evaluated_depsgraph_get()

    bindings = []
    confidences = []
    mask = []
    for point in request["points"]:
        if point is None:
            bindings.append(None)
            confidences.append(0.0)
            mask.append(False)
            continue
        xyz = Vector(point["xyz"])
        depsgraph = use_frame(point["frame"])
        votes = Counter()
        air = 0
        for origin, direction in zip(point["origins"], point["directions"]):
            hit, _, _, _, obj, _ = scene.ray_cast(depsgraph, Vector(origin), Vector(direction))
            if hit:
                votes[obj.name] += 1
            else:
                air += 1
        ranking = votes.most_common(2)
        name, count = ranking[0] if ranking else (None, 0)
        second = ranking[1][1] if len(ranking) > 1 else 0
        n_rays = len(point["origins"])
        if name is None or count <= air or count <= second:
            bindings.append(("air", point["xyz"]))
            confidences.append((n_rays - count) / n_rays)
        else:
            obj = bpy.data.objects[name].evaluated_get(depsgraph)
            ok, location, _, _ = obj.closest_point_on_mesh(obj.matrix_world.inverted() @ xyz)
            if ok:
                bindings.append(("object", name, [location.x, location.y, location.z]))
                confidences.append(count / n_rays)
            else:
                bindings.append(("air", point["xyz"]))
                confidences.append((n_rays - count) / n_rays)
        mask.append(True)

    worlds = []
    visibility = []
    for frame_index, centers in enumerate(request["centers"]):
        depsgraph = use_frame(frame_index)
        frame_worlds = []
        for binding in bindings:
            if binding is None:
                frame_worlds.append([0.0, 0.0, 0.0])
            elif binding[0] == "air":
                frame_worlds.append(binding[1])
            else:
                world = bpy.data.objects[binding[1]].evaluated_get(depsgraph).matrix_world @ Vector(binding[2])
                frame_worlds.append([world.x, world.y, world.z])
        worlds.append(frame_worlds)
        view_visibility = []
        for center in centers:
            origin = Vector(center)
            flags = []
            for binding, world in zip(bindings, frame_worlds):
                if binding is None:
                    flags.append(0.0)
                    continue
                delta = Vector(world) - origin
                distance = delta.length
                if distance == 0.0:
                    flags.append(1.0)
                    continue
                direction = delta.normalized()
                hit, location, _, _, _, _ = scene.ray_cast(depsgraph, origin, direction)
                flags.append(0.0 if hit and (location - origin).dot(direction) + eps < distance else 1.0)
            view_visibility.append(flags)
        visibility.append(view_visibility)
        if frame_index % 25 == 0:
            print(f"tracked frame {frame_index + 1}/{len(request['centers'])}", flush=True)

    args.output.write_text(json.dumps({
        "worlds": worlds,
        "visibility": visibility,
        "confidence": confidences,
        "mask": mask,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
