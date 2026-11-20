# run main.py to generate the dataset with different HDRI images already set up and 3D object saved

import blenderproc as bproc
import numpy as np
import random
import os

bproc.init()
bproc.renderer.set_max_amount_of_samples(32)
bproc.renderer.set_render_devices(use_only_cpu=False)

# path to 3D object
objs = bproc.loader.load_obj("/path/to/object") # set the path for the 3D object; for example: "/Users/user/Downloads/sunglasses.obj"
obj = objs[0]

obj.set_location([0, 0, 0])

# path to hdri directory
hdri_dir = "/path/to/hdri/directory" # set the path for the directory with HDRIs; for example: "/Users/user/Desktop/Custom_Object_Detection/hdris/"
hdri_files = [os.path.join(hdri_dir, f"{i:02d}.hdr") for i in range(11)]

# camera intrinsics
bproc.camera.set_resolution(640, 640)

bbox = obj.get_bound_box()
max_dim = max([max(abs(v[0]), abs(v[1]), abs(v[2])) for v in bbox])
base_camera_distance = max_dim * 2.5

print(f"Object max dimension: {max_dim}, Base camera distance: {base_camera_distance}")

for hdri_idx, hdri_path in enumerate(hdri_files):
    print(f"Rendering with HDRI {hdri_idx + 1}/11: {hdri_path}")

    bproc.world.set_world_background_hdr_img(hdri_path)
    
    for i in range(91):
        obj.set_rotation_euler(np.random.uniform(0, 2*np.pi, size=3))
        obj_offset = np.random.uniform(-0.02, 0.02, size=3)
        obj.set_location(obj_offset)
        camera_distance = np.random.uniform(2, 15)
        
        location = bproc.sampler.sphere(
            center=obj_offset,
            radius=camera_distance,
            mode="SURFACE"
        )
        
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            forward_vec=obj_offset - location,
            inplane_rot=np.random.uniform(0, 2*np.pi)
        )
        
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    # adding lights
    num_lights = random.randint(1, 5)
    for _ in range(num_lights):
        light = bproc.types.Light()
        light.set_type(random.choice(["POINT", "SUN", "AREA"]))
        light.set_location(bproc.sampler.shell(
            center=[0, 0, 0],
            radius_min=2,
            radius_max=4,
            elevation_min=5,
            elevation_max=89
        ))
        light.set_energy(np.random.uniform(20, 75))
    
    # enable segmentation BEFORE rendering each batch
    bproc.renderer.enable_segmentation_output(
        map_by=["instance", "class"],
        default_values={'category_id': 1}
    )
    
    print(f"Rendering {91} images with this HDRI...")
    data = bproc.renderer.render()
    
    # save with COCO annotations
    bproc.writer.write_coco_annotations(
        output_dir="/path/to/blenderproc_output", # set output directory where annotations are saved; for example "/Users/user/Desktop/Custom_Object_Detection/blenderproc_output"
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        colors=data["colors"],
        append_to_existing_output=(hdri_idx > 0)
    )
    
    print(f"Batch {hdri_idx + 1} complete")
    bproc.utility.reset_keyframes()

print("Done! All HDRIs processed with COCO annotations")