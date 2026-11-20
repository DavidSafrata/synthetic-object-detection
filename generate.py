import blenderproc as bproc
import argparse
import json
import os
import bpy
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("split", choices=["train", "val", "test"])
parser.add_argument("n", type=int)
parser.add_argument("out")
parser.add_argument("--focus", help="failures.json written by evaluate.py")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

cfg = yaml.safe_load(open("config.yaml"))
rng = np.random.default_rng(args.seed)
size = cfg["object_size"]
fov = np.radians(cfg["fov"])

bproc.init()
bproc.renderer.set_max_amount_of_samples(cfg["samples"])
bproc.camera.set_intrinsics_from_blender_params(lens=fov, image_width=cfg["resolution"], image_height=cfg["resolution"], lens_unit="FOV")

obj = bproc.loader.load_obj(cfg["object"])[0]
bbox = obj.get_bound_box()
obj.set_origin(point=bbox.mean(axis=0))
obj.set_scale([size / (bbox.max(axis=0) - bbox.min(axis=0)).max()] * 3)
obj.persist_transformation_into_mesh(location=False)
obj.set_cp("category_id", 1)

occluders = []
for shape in ["CUBE", "CYLINDER", "SPHERE", "CONE"]:
    o = bproc.object.create_primitive(shape)
    o.replace_materials(bproc.material.create(shape))
    occluders.append(o)
lights = [bproc.types.Light() for _ in range(cfg["max_lights"])]
background = [n for n in bpy.context.scene.world.node_tree.nodes if n.type == "BACKGROUND"][0]
bproc.renderer.enable_segmentation_output(map_by=["instance", "class"], default_values={"category_id": 0})

# each factor is split into bins; --focus shifts the bin probabilities towards bins where the detector failed
factors = cfg["factors"]
weights = {k: np.ones(len(v) - 1) / (len(v) - 1) for k, v in factors.items()}
if args.focus:
    failures = json.load(open(args.focus))
    for k in weights:
        miss = np.array(failures[k]["miss_rate"])
        weights[k] = (1 - cfg["focus_mix"]) * weights[k] + cfg["focus_mix"] * miss / miss.sum()
        print(k, np.round(weights[k], 2))

meta_path = os.path.join(args.out, "meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else []
remaining = args.n

while remaining > 0:
    frames = min(cfg["frames_per_batch"], remaining)
    hdri = rng.choice(cfg["hdris"][args.split])
    bproc.world.set_world_background_hdr_img(os.path.join(cfg["hdri_dir"], f"{hdri}.hdr"), rotation_euler=[0, 0, rng.uniform(0, 2 * np.pi)])
    for m in obj.get_materials():
        m.set_principled_shader_value("Roughness", rng.uniform(*cfg["roughness"]))
    for o in occluders:
        m = o.get_materials()[0]
        m.set_principled_shader_value("Base Color", [*rng.uniform(0, 1, 3), 1])
        m.set_principled_shader_value("Roughness", rng.uniform(0, 1))
        m.set_principled_shader_value("Metallic", float(rng.random() < 0.3))

    for i in range(frames):
        f = {}
        for k, edges in factors.items():
            b = rng.choice(len(weights[k]), p=weights[k])
            f[k] = float(rng.uniform(edges[b], edges[b + 1]))
        f["occluders"] = int(f["occluders"])
        negative = bool(rng.random() < cfg["negative_ratio"])

        el, az = np.radians(f["elevation"]), rng.uniform(0, 2 * np.pi)
        cam = f["distance"] * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        target = rng.uniform(-1, 1, 3) * cfg["offcenter"] * f["distance"] * np.tan(fov / 2)
        rotation = bproc.camera.rotation_from_forward_vec(target - cam, inplane_rot=rng.uniform(0, 2 * np.pi))
        bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(cam, rotation), frame=i)

        # hidden objects are moved behind the camera
        obj.set_location(2 * cam if negative else [0, 0, 0], frame=i)
        obj.set_rotation_euler([*rng.uniform(-0.3, 0.3, 2), rng.uniform(0, 2 * np.pi)], frame=i)
        for j, idx in enumerate(rng.permutation(len(occluders))):
            t = rng.uniform(0.15, 0.6)
            visible = j < f["occluders"]
            occluders[idx].set_location(cam * t + rng.normal(0, 0.35 * size, 3) if visible else 2 * cam, frame=i)
            occluders[idx].set_rotation_euler(rng.uniform(0, 2 * np.pi, 3), frame=i)
            occluders[idx].set_scale(rng.uniform(0.1, 0.3, 3) * size, frame=i)

        for light in lights:
            light.set_location(rng.normal(0, 1, 3) * rng.uniform(0.5, 2), frame=i)
            light.set_energy(rng.uniform(5, 60) if rng.random() < 0.5 else 0, frame=i)
        background.inputs["Strength"].default_value = f["brightness"]
        background.inputs["Strength"].keyframe_insert("default_value", frame=i)

        meta.append({"file": f"images/{len(meta):06d}.jpg", "hdri": str(hdri), "negative": negative, **f})

    data = bproc.renderer.render()
    bproc.writer.write_coco_annotations(args.out, instance_segmaps=data["instance_segmaps"], instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"], color_file_format="JPEG", jpg_quality=90, append_to_existing_output=True)
    json.dump(meta, open(meta_path, "w"), indent=1)
    bproc.utility.reset_keyframes()
    remaining -= frames
    print(f"{len(meta)} images in {args.out}")
